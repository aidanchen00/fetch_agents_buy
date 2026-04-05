"""
Search Agent — searches Amazon for candidate products via Browser Use.

Responsibilities:
- Receives SearchRequest from orchestrator
- Uses Browser Use Cloud to search Amazon for each item
- Returns structured SearchResults to orchestrator
- Posts session/event metadata to FastAPI
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
from uagents import Agent, Context, Protocol

from agents.shared.browser_service import get_browser_use_service
from agents.shared.config import (
    FASTAPI_CALLBACK_URL,
    SEARCH_PORT,
    SEARCH_SEED,
)
from agents.shared.messages import SearchRequest, SearchResults

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("search_agent")

# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

search_agent = Agent(
    name="search-agent",
    seed=SEARCH_SEED,
    port=SEARCH_PORT,
    mailbox=True,
)

search_proto = Protocol(name="search-protocol")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def post_event(run_id: str, event_type: str, payload: dict):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(FASTAPI_CALLBACK_URL, json={
                "run_id": run_id,
                "agent_name": "search",
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "payload": payload,
            })
    except Exception as e:
        logger.warning(f"Failed to post event {event_type}: {e}")


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

@search_proto.on_message(SearchRequest)
async def handle_search_request(ctx: Context, sender: str, msg: SearchRequest):
    logger.info(f"[search] Searching for {len(msg.items)} items for run {msg.run_id[:8]}")

    await post_event(msg.run_id, "search_started", {"item_count": len(msg.items)})

    svc = get_browser_use_service()
    all_results: dict = {}

    for item_dict in msg.items:
        item_name = item_dict.get("name", "unknown")
        search_query = item_dict.get("search_query", item_name)
        logger.info(f"[search] Searching for: '{search_query}'")

        async def _on_session(sid: str, url: str, _item=item_name):
            await post_event(msg.run_id, "session_created", {
                "session_id": sid,
                "live_view_url": url,
                "debugger_url": "",
                "item": _item,
            })

        candidates = []
        try:
            result = await svc.search_amazon(
                search_query,
                max_results=8,
                on_session_created=_on_session,
            )
            candidates = result.candidates
            logger.info(f"[search] Found {len(candidates)} candidates for '{item_name}'")

        except Exception as e:
            logger.error(f"[search] Error searching for '{item_name}': {e}")
            candidates = []

        all_results[item_name] = candidates

        await post_event(msg.run_id, "item_searched", {
            "item_name": item_name,
            "candidate_count": len(candidates),
            "query": search_query,
        })

    await post_event(msg.run_id, "search_complete", {
        "items_searched": len(all_results),
        "total_candidates": sum(len(v) for v in all_results.values()),
    })

    response = SearchResults(run_id=msg.run_id, results=all_results)
    await ctx.send(sender, response)
    logger.info(f"[search] Sent SearchResults for run {msg.run_id[:8]}")


search_agent.include(search_proto)

if __name__ == "__main__":
    search_agent.run()
