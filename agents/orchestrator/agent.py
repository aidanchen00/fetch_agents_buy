"""
Orchestrator Agent — the ASI:One-facing entry point.

Responsibilities:
- ASI:One-compatible chat interface (chat protocol)
- Polls SQLite every 2s for new web-triggered runs
- Parses shopping instructions into structured items
- Coordinates search → rank → budget → buy pipeline
- Aggregates results and reports back via SSE/SQLite
- Maintains in-memory run state for concurrent runs

ASI:One integration:
  - mailbox=True: Agentverse stores messages when agent is offline
  - publish_agent_details=True: visible in Agentverse search
  - readme_path="README.md": shown in Agentverse listing
  - chat_protocol_spec: required for ASI:One compatibility
"""
from __future__ import annotations

import asyncio
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (  # type: ignore
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

from agents.shared.config import (
    BUYER_A_PORT, BUYER_A_SEED,
    BUYER_B_PORT, BUYER_B_SEED,
    BUYER_C_PORT, BUYER_C_SEED,
    BUYER_D_PORT, BUYER_D_SEED,
    BUYER_E_PORT, BUYER_E_SEED,
    DATABASE_PATH,
    FASTAPI_CALLBACK_URL,
    ORCHESTRATOR_PORT,
    ORCHESTRATOR_SEED,
    RANKER_PORT, RANKER_SEED,
    SEARCH_PORT, SEARCH_SEED,
    TOTAL_BUDGET,
    TREASURY_PORT, TREASURY_SEED,
)
from agents.shared.messages import (
    BudgetRequest,
    BudgetResponse,
    BuyRequest,
    BuyResultMsg,
    RankRequest,
    RankResult,
    SearchRequest,
    SearchResults,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("orchestrator")

# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

orchestrator = Agent(
    name="shopping-orchestrator",
    seed=ORCHESTRATOR_SEED,
    port=ORCHESTRATOR_PORT,
    mailbox=True,
    publish_agent_details=True,
    readme_path=str(Path(__file__).parent / "README.md"),
)


# ---------------------------------------------------------------------------
# Helper: derive addresses from seeds
# ---------------------------------------------------------------------------

def _addr(seed: str) -> str:
    try:
        from uagents.crypto import Identity  # type: ignore
        return Identity.from_seed(seed, 0).address
    except Exception:
        return f"placeholder-{seed[:8]}"


SEARCH_ADDR = _addr(SEARCH_SEED)
RANKER_ADDR = _addr(RANKER_SEED)
TREASURY_ADDR = _addr(TREASURY_SEED)
BUYER_ADDRS = [
    _addr(BUYER_A_SEED),
    _addr(BUYER_B_SEED),
    _addr(BUYER_C_SEED),
    _addr(BUYER_D_SEED),
    _addr(BUYER_E_SEED),
]

logger.info(f"Orchestrator address: {orchestrator.address}")
logger.info(f"Search agent address: {SEARCH_ADDR}")
logger.info(f"Ranker agent address: {RANKER_ADDR}")


# ---------------------------------------------------------------------------
# In-memory run state
# ---------------------------------------------------------------------------

class RunState:
    """Tracks pipeline progress for a single run."""

    def __init__(self, run_id: str, instruction: str, items: List[dict]):
        self.run_id = run_id
        self.instruction = instruction
        self.items = items  # List[ShoppingItem dicts]
        self.search_results: Dict[str, List[dict]] = {}
        self.rank_results: Dict[str, dict] = {}       # item_name -> chosen candidate
        self.rank_reasons: Dict[str, str] = {}
        self.budget_approvals: Dict[str, dict] = {}   # item_name -> approval
        self.buy_results: Dict[str, dict] = {}        # item_name -> buy result
        self.asi_one_sender: Optional[str] = None     # set if triggered from ASI:One
        self.pending_ranks = 0
        self.pending_buys = 0
        self.ctx_ref: Optional[Context] = None        # stored for async sends

    @property
    def all_ranked(self) -> bool:
        return self.pending_ranks == 0 and len(self.rank_results) == len(self.items)

    @property
    def all_bought(self) -> bool:
        return self.pending_buys == 0 and len(self.buy_results) == len(self.items)


# Global state store: run_id -> RunState
_run_states: Dict[str, RunState] = {}


# ---------------------------------------------------------------------------
# Instruction parser
# ---------------------------------------------------------------------------

def parse_instruction(instruction: str) -> List[dict]:
    """
    Parse a natural-language shopping instruction into structured items.

    Handles patterns like:
      "AA batteries, under $18, quantity 2"
      "1. USB-C charger 65W, max price $30, qty 1"
      "Buy: notebooks under $8 each, 3"
    """
    items = []
    lines = re.split(r"\n|(?<=\d)\.\s+", instruction.strip())
    if len(lines) == 1:
        lines = re.split(r";\s*", instruction)
    if len(lines) == 1:
        lines = re.split(
            r",\s+(?=\w+.*(?:\$[\d]|under|max|below|quantity|qty))",
            instruction,
            flags=re.IGNORECASE,
        )

    for line in lines:
        line = line.strip().strip(".,;").strip()
        if not line or len(line) < 3:
            continue

        # Extract max price
        price_match = re.search(r"(?:under|max|below|less than|up to)?\s*\$?([\d]+(?:\.\d{1,2})?)\s*(?:each)?", line, re.IGNORECASE)
        max_price = float(price_match.group(1)) if price_match else 50.0

        # Extract quantity
        qty_match = re.search(r"(?:quantity|qty|x|×)\s*(\d+)|(\d+)\s*(?:units?|packs?|pieces?|count|ct)", line, re.IGNORECASE)
        if not qty_match:
            qty_match = re.search(r"\b(\d+)\s*$", line)
        qty = int((qty_match.group(1) or qty_match.group(2) or "1").strip()) if qty_match else 1

        # Extract item name (remove price/qty tokens)
        name_line = re.sub(r"(?:under|max|below|less than|up to)?\s*\$[\d]+(?:\.\d{1,2})?", "", line, flags=re.IGNORECASE)
        name_line = re.sub(r"(?:quantity|qty)\s*\d+", "", name_line, flags=re.IGNORECASE)
        name_line = re.sub(r"\b\d+\s*(?:units?|packs?|pieces?|count|ct)\b", "", name_line, flags=re.IGNORECASE)
        name_line = re.sub(r"^\d+\.\s*", "", name_line)  # Remove list numbering
        name_line = re.sub(r"\beach\b", "", name_line, flags=re.IGNORECASE)
        # Strip leading action verbs ("buy", "get", "order", "purchase")
        name_line = re.sub(r"^(?:buy|get|order|purchase|add)\s+", "", name_line, flags=re.IGNORECASE)
        name = re.sub(r"\s+", " ", name_line).strip().strip(".,;:")

        if not name or len(name) < 2:
            continue

        # Build search query (clean version of name)
        search_query = name

        items.append({
            "name": name,
            "max_price": max_price,
            "quantity": max(1, qty),
            "search_query": search_query,
        })

    return items if items else [{
        "name": instruction[:60],
        "max_price": 50.0,
        "quantity": 1,
        "search_query": instruction[:60],
    }]


# ---------------------------------------------------------------------------
# FastAPI event reporting
# ---------------------------------------------------------------------------

async def post_event(run_id: str, event_type: str, payload: dict):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(FASTAPI_CALLBACK_URL, json={
                "run_id": run_id,
                "agent_name": "orchestrator",
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "payload": payload,
            })
    except Exception as e:
        logger.warning(f"Failed to post event {event_type}: {e}")


# ---------------------------------------------------------------------------
# SQLite polling helpers
# ---------------------------------------------------------------------------

def _get_pending_run():
    """Fetch one pending run from SQLite. Returns a dict or None."""
    try:
        import sqlite3
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT id, instruction, total_budget FROM runs WHERE status='pending' ORDER BY created_at LIMIT 1"
        )
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.debug(f"SQLite poll error: {e}")
        return None


def _mark_run_in_progress(run_id: str):
    try:
        import sqlite3
        from datetime import datetime as dt
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute(
            "UPDATE runs SET status='in_progress', updated_at=? WHERE id=?",
            (dt.utcnow().isoformat(), run_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not mark run in_progress: {e}")


def _update_run_complete(run_id: str, results: list, total_spent: float):
    try:
        import json
        import sqlite3
        from datetime import datetime as dt
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute(
            "UPDATE runs SET status='completed', results=?, total_spent=?, updated_at=? WHERE id=?",
            (json.dumps(results), total_spent, dt.utcnow().isoformat(), run_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not mark run complete: {e}")


def _update_run_failed(run_id: str, error: str):
    try:
        import sqlite3
        from datetime import datetime as dt
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute(
            "UPDATE runs SET status='failed', updated_at=? WHERE id=?",
            (dt.utcnow().isoformat(), run_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not mark run failed: {e}")


def _update_run_items(run_id: str, items: list):
    try:
        import json
        import sqlite3
        from datetime import datetime as dt
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute(
            "UPDATE runs SET items=?, updated_at=? WHERE id=?",
            (json.dumps(items), dt.utcnow().isoformat(), run_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not update run items: {e}")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

async def _start_pipeline(ctx: Context, run_id: str, instruction: str, total_budget: float):
    """Parse instruction and kick off search."""
    items = parse_instruction(instruction)
    logger.info(f"[orchestrator] Parsed {len(items)} items from instruction")

    state = RunState(run_id=run_id, instruction=instruction, items=items)
    state.ctx_ref = ctx
    _run_states[run_id] = state

    _update_run_items(run_id, items)

    await post_event(run_id, "run_started", {
        "instruction": instruction[:100],
        "item_count": len(items),
        "items": items,
    })

    await post_event(run_id, "parsing_done", {"items": items})

    # Send search request
    search_req = SearchRequest(run_id=run_id, items=items)
    await ctx.send(SEARCH_ADDR, search_req)
    logger.info(f"[orchestrator] Sent SearchRequest to {SEARCH_ADDR[:20]}...")


async def _finalize_run(ctx: Context, state: RunState):
    """All buys complete — aggregate results and report."""
    results = list(state.buy_results.values())
    total_spent = sum(
        float(r.get("final_price", 0)) * int(r.get("quantity", 1))
        for r in results
        if r.get("status") == "success"
    )

    _update_run_complete(state.run_id, results, total_spent)

    summary_lines = []
    for r in results:
        status = "✓" if r.get("status") == "success" else "✗"
        name = r.get("item_name", "?")
        price = r.get("final_price", 0)
        summary_lines.append(f"{status} {name}: ${price:.2f}")

    summary = "\n".join(summary_lines) or "No items processed"

    await post_event(state.run_id, "run_complete", {
        "results": results,
        "total_spent": total_spent,
        "summary": summary,
        "success_count": sum(1 for r in results if r.get("status") == "success"),
        "fail_count": sum(1 for r in results if r.get("status") != "success"),
    })

    logger.info(f"[orchestrator] Run {state.run_id[:8]} complete. Spent ${total_spent:.2f}")

    # Reply to ASI:One if this was chat-triggered
    if state.asi_one_sender:
        reply_text = f"Shopping complete!\n\n{summary}\n\nTotal spent: ${total_spent:.2f}"
        await ctx.send(
            state.asi_one_sender,
            ChatMessage(
                timestamp=datetime.utcnow(),
                msg_id=uuid4(),
                content=[
                    TextContent(type="text", text=reply_text),
                    EndSessionContent(type="end-session"),
                ],
            )
        )

    # Cleanup state
    del _run_states[state.run_id]


# ---------------------------------------------------------------------------
# Chat protocol (ASI:One interface)
# ---------------------------------------------------------------------------

chat_proto = Protocol(spec=chat_protocol_spec)


@chat_proto.on_message(ChatMessage)
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat messages from ASI:One."""
    # Always acknowledge first
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.utcnow(),
            acknowledged_msg_id=msg.msg_id,
        )
    )

    # Extract text content
    instruction = ""
    for item in msg.content:
        if isinstance(item, TextContent):
            instruction += item.text + " "
    instruction = instruction.strip()

    logger.info(f"[orchestrator] ASI:One message from {sender[:20]}...: {instruction[:80]}")

    if not instruction:
        await ctx.send(
            sender,
            ChatMessage(
                timestamp=datetime.utcnow(),
                msg_id=uuid4(),
                content=[
                    TextContent(type="text", text="Please send a shopping instruction. Example: 'Buy AA batteries under $18, quantity 2'"),
                    EndSessionContent(type="end-session"),
                ],
            )
        )
        return

    # Create a run in SQLite for this chat request
    import sqlite3
    import json as _json
    from datetime import datetime as dt
    run_id = str(uuid4())
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute(
            "INSERT INTO runs (id, instruction, status, items, results, total_budget, total_spent, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, instruction, "in_progress", "[]", "[]", TOTAL_BUDGET, 0.0,
             dt.utcnow().isoformat(), dt.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to create run in SQLite for ASI:One request: {e}")

    # Start pipeline
    await _start_pipeline(ctx, run_id, instruction, TOTAL_BUDGET)

    # Store sender for final reply
    if run_id in _run_states:
        _run_states[run_id].asi_one_sender = sender

    # Send immediate acknowledgement to user
    await ctx.send(
        sender,
        ChatMessage(
            timestamp=datetime.utcnow(),
            msg_id=uuid4(),
            content=[
                TextContent(type="text", text=f"Processing your shopping request (run_id: {run_id[:8]}...). I'll report back when complete."),
            ],
        )
    )


@chat_proto.on_message(ChatAcknowledgement)
async def handle_chat_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    pass  # No action needed for acks


orchestrator.include(chat_proto, publish_manifest=True)


# ---------------------------------------------------------------------------
# Internal protocol (worker agent responses)
# ---------------------------------------------------------------------------

internal_proto = Protocol(name="orchestrator-internal")


@internal_proto.on_message(SearchResults)
async def on_search_results(ctx: Context, sender: str, msg: SearchResults):
    """Receive search results; fan out to ranker for each item."""
    state = _run_states.get(msg.run_id)
    if not state:
        logger.warning(f"[orchestrator] Received SearchResults for unknown run {msg.run_id[:8]}")
        return

    state.ctx_ref = ctx
    state.search_results = msg.results
    logger.info(f"[orchestrator] Search complete for run {msg.run_id[:8]}: {len(msg.results)} items")

    # Fan out rank requests — one per item
    items_with_candidates = 0
    for item in state.items:
        item_name = item["name"]
        candidates = msg.results.get(item_name, [])

        if not candidates:
            logger.warning(f"[orchestrator] No candidates for '{item_name}', skipping")
            # Mark as failed rank
            state.rank_results[item_name] = {}
            state.rank_reasons[item_name] = "No search results"
            continue

        items_with_candidates += 1
        state.pending_ranks += 1
        rank_req = RankRequest(run_id=msg.run_id, item=item, candidates=candidates)
        await ctx.send(RANKER_ADDR, rank_req)
        logger.info(f"[orchestrator] Sent RankRequest for '{item_name}' ({len(candidates)} candidates)")

    if items_with_candidates == 0:
        # All items failed search — mark run failed
        await post_event(msg.run_id, "run_failed", {"reason": "All item searches returned no candidates"})
        _update_run_failed(msg.run_id, "All item searches failed")
        del _run_states[msg.run_id]


@internal_proto.on_message(RankResult)
async def on_rank_result(ctx: Context, sender: str, msg: RankResult):
    """Collect rank results; when all done, request budget approval."""
    state = _run_states.get(msg.run_id)
    if not state:
        return

    state.ctx_ref = ctx
    state.rank_results[msg.item_name] = msg.chosen
    state.rank_reasons[msg.item_name] = msg.reason
    state.pending_ranks = max(0, state.pending_ranks - 1)

    logger.info(f"[orchestrator] Rank result for '{msg.item_name}': {msg.chosen.get('title', '')[:40]}")

    # Wait until all ranks are complete (or all items accounted for)
    items_ranked = len(state.rank_results)
    if items_ranked < len(state.items):
        return  # still waiting

    # All ranked — request budget
    logger.info(f"[orchestrator] All {len(state.items)} items ranked. Requesting budget approval.")

    line_items = []
    for item in state.items:
        item_name = item["name"]
        chosen = state.rank_results.get(item_name, {})
        price = float(chosen.get("price", item.get("max_price", 50)))
        qty = item.get("quantity", 1)
        line_items.append({
            "item_name": item_name,
            "amount": price * qty,
            "quantity": qty,
            "unit_price": price,
        })

    total = sum(li["amount"] for li in line_items)
    budget_req = BudgetRequest(run_id=msg.run_id, line_items=line_items, total_amount=total)
    await ctx.send(TREASURY_ADDR, budget_req)
    logger.info(f"[orchestrator] Sent BudgetRequest: total=${total:.2f}")


@internal_proto.on_message(BudgetResponse)
async def on_budget_response(ctx: Context, sender: str, msg: BudgetResponse):
    """Receive budget approvals; dispatch buy requests to buyer agents in parallel."""
    state = _run_states.get(msg.run_id)
    if not state:
        return

    state.ctx_ref = ctx

    # Index approvals by item name
    for approval in msg.approvals:
        state.budget_approvals[approval["item_name"]] = approval

    logger.info(f"[orchestrator] Budget: approved=${msg.approved_total:.2f}, denied={msg.denied_items}")

    # Assign buyer agents round-robin
    approved_items = [
        item for item in state.items
        if state.budget_approvals.get(item["name"], {}).get("approved", False)
        and state.rank_results.get(item["name"])
    ]

    if not approved_items:
        logger.warning(f"[orchestrator] No approved items to buy for run {msg.run_id[:8]}")
        await post_event(msg.run_id, "run_failed", {"reason": "All items denied by treasury"})
        _update_run_failed(msg.run_id, "All items denied by treasury")
        del _run_states[msg.run_id]
        return

    state.pending_buys = len(approved_items)

    for i, item in enumerate(approved_items):
        item_name = item["name"]
        buyer_addr = BUYER_ADDRS[i % len(BUYER_ADDRS)]
        approval = state.budget_approvals[item_name]
        chosen = state.rank_results[item_name]

        buy_req = BuyRequest(
            run_id=msg.run_id,
            item=item,
            chosen_product=chosen,
            approval_ref=approval["reference_id"],
            quantity=item.get("quantity", 1),
        )
        await ctx.send(buyer_addr, buy_req)
        logger.info(f"[orchestrator] Dispatched buy for '{item_name}' to buyer {i % len(BUYER_ADDRS) + 1}")

        await post_event(msg.run_id, "buy_dispatched", {
            "item_name": item_name,
            "buyer": f"buyer_{chr(ord('a') + (i % len(BUYER_ADDRS)))}",
            "product_title": chosen.get("title", "")[:60],
            "price": chosen.get("price", 0),
        })

    # Handle denied items
    for item in state.items:
        item_name = item["name"]
        approval = state.budget_approvals.get(item_name, {})
        if not approval.get("approved", False):
            state.buy_results[item_name] = {
                "run_id": msg.run_id,
                "item_name": item_name,
                "status": "skipped",
                "final_price": 0.0,
                "quantity": item.get("quantity", 1),
                "screenshot_url": "",
                "screenshot_path": "",
                "live_view_url": "",
                "session_id": "",
                "agent_name": "treasury",
                "error": approval.get("denial_reason", "Budget denied"),
            }
            state.pending_buys = max(0, state.pending_buys - 0)  # already accounted for above

    # If all items were denied, finalize immediately
    if state.pending_buys == 0:
        await _finalize_run(ctx, state)


@internal_proto.on_message(BuyResultMsg)
async def on_buy_result(ctx: Context, sender: str, msg: BuyResultMsg):
    """Collect buy results; finalize when all done."""
    state = _run_states.get(msg.run_id)
    if not state:
        logger.warning(f"[orchestrator] BuyResultMsg for unknown run {msg.run_id[:8]}")
        return

    state.ctx_ref = ctx
    state.buy_results[msg.item_name] = {
        "run_id": msg.run_id,
        "item_name": msg.item_name,
        "status": msg.status,
        "final_price": msg.final_price,
        "quantity": msg.quantity,
        "screenshot_url": msg.screenshot_url,
        "screenshot_path": msg.screenshot_path,
        "live_view_url": msg.live_view_url,
        "session_id": msg.session_id,
        "agent_name": msg.agent_name,
        "error": msg.error,
    }
    state.pending_buys = max(0, state.pending_buys - 1)

    logger.info(f"[orchestrator] Buy result for '{msg.item_name}': {msg.status} (pending={state.pending_buys})")

    # Check if all buys are complete
    all_items_accounted = len(state.buy_results) == len(state.items)
    if all_items_accounted and state.pending_buys == 0:
        await _finalize_run(ctx, state)


orchestrator.include(internal_proto)


# ---------------------------------------------------------------------------
# SQLite polling interval
# ---------------------------------------------------------------------------

@orchestrator.on_interval(period=2.0)
async def poll_for_pending_runs(ctx: Context):
    """Check SQLite for new web-triggered runs every 2 seconds."""
    row = _get_pending_run()
    if not row:
        return

    run_id = row["id"]
    instruction = row["instruction"]
    total_budget = float(row.get("total_budget", TOTAL_BUDGET))

    logger.info(f"[orchestrator] Picked up pending run {run_id[:8]}: '{instruction[:60]}'")
    _mark_run_in_progress(run_id)

    try:
        await _start_pipeline(ctx, run_id, instruction, total_budget)
    except Exception as e:
        logger.error(f"[orchestrator] Pipeline start failed for {run_id[:8]}: {e}")
        await post_event(run_id, "run_failed", {"reason": str(e)})
        _update_run_failed(run_id, str(e))


# ---------------------------------------------------------------------------
# Startup log
# ---------------------------------------------------------------------------

@orchestrator.on_event("startup")
async def on_startup(ctx: Context):
    logger.info(f"[orchestrator] Started. Address: {orchestrator.address}")
    logger.info(f"[orchestrator] Polling SQLite at {DATABASE_PATH}")
    logger.info(f"[orchestrator] Posting events to {FASTAPI_CALLBACK_URL}")


if __name__ == "__main__":
    orchestrator.run()
