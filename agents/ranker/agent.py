"""
Ranker Agent — selects the best candidate product for a shopping item.

Scoring criteria:
1. Must be under max_price (hard filter)
2. Keyword overlap with item name (weighted)
3. Rating (weighted)
4. Review count (weighted)
5. Price-to-budget ratio (prefer cheaper within budget)

Returns a ranked selection with a human-readable reason.
"""
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
from uagents import Agent, Context, Protocol

from agents.shared.config import FASTAPI_CALLBACK_URL, RANKER_PORT, RANKER_SEED
from agents.shared.messages import RankRequest, RankResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ranker_agent")

# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

ranker_agent = Agent(
    name="ranker-agent",
    seed=RANKER_SEED,
    port=RANKER_PORT,
    endpoint=[f"http://localhost:{RANKER_PORT}/submit"],
)

ranker_proto = Protocol(name="ranker-protocol")


# ---------------------------------------------------------------------------
# Ranking logic
# ---------------------------------------------------------------------------

def _keyword_score(title: str, item_name: str) -> float:
    """Fraction of item keywords found in candidate title."""
    title_lower = title.lower()
    keywords = [w for w in item_name.lower().split() if len(w) > 2]
    if not keywords:
        return 0.5
    hits = sum(1 for kw in keywords if kw in title_lower)
    return hits / len(keywords)


def _rank_candidates(
    item_name: str,
    max_price: float,
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Score and rank candidates. Returns best candidate dict, or None."""
    eligible = [c for c in candidates if 0 < float(c.get("price", 0)) <= max_price]

    if not eligible:
        # Relax to slightly over budget (5% margin)
        eligible = [c for c in candidates if 0 < float(c.get("price", 0)) <= max_price * 1.05]

    if not eligible:
        return None

    def score(c: Dict[str, Any]) -> float:
        price = float(c.get("price", max_price))
        rating = float(c.get("rating") or 0)
        review_count = int(c.get("review_count") or 0)
        title = str(c.get("title", ""))

        kw_score = _keyword_score(title, item_name)  # 0-1
        price_score = 1 - (price / max_price)         # 0-1 (lower price is better)
        rating_score = rating / 5.0                    # 0-1
        popularity = min(1.0, review_count / 1000)     # 0-1, capped at 1000 reviews

        # Weights: keyword match is most important, then rating, then price, then popularity
        return (
            kw_score * 0.40 +
            rating_score * 0.30 +
            price_score * 0.20 +
            popularity * 0.10
        )

    ranked = sorted(eligible, key=score, reverse=True)
    return ranked[0]


def _build_reason(chosen: Dict[str, Any], item_name: str, max_price: float) -> str:
    price = chosen.get("price", 0)
    rating = chosen.get("rating")
    title = chosen.get("title", "")[:60]
    kw = _keyword_score(title, item_name)

    parts = [f"Best match for '{item_name}'"]
    if kw >= 0.6:
        parts.append("strong keyword match")
    elif kw >= 0.3:
        parts.append("partial keyword match")
    if rating:
        parts.append(f"rated {rating:.1f}/5")
    parts.append(f"${price:.2f} (budget ${max_price:.2f})")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def post_event(run_id: str, event_type: str, payload: dict):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(FASTAPI_CALLBACK_URL, json={
                "run_id": run_id,
                "agent_name": "ranker",
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "payload": payload,
            })
    except Exception as e:
        logger.warning(f"Failed to post event: {e}")


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

@ranker_proto.on_message(RankRequest)
async def handle_rank_request(ctx: Context, sender: str, msg: RankRequest):
    item_name = msg.item.get("name", "unknown")
    max_price = float(msg.item.get("max_price", 9999))
    logger.info(f"[ranker] Ranking {len(msg.candidates)} candidates for '{item_name}' (max ${max_price})")

    chosen = _rank_candidates(item_name, max_price, msg.candidates)

    if not chosen:
        logger.warning(f"[ranker] No eligible candidates for '{item_name}' under ${max_price}")
        await post_event(msg.run_id, "ranking_failed", {
            "item_name": item_name,
            "reason": f"No candidates under ${max_price}",
            "candidate_count": len(msg.candidates),
        })
        # Send back a "failed" result with empty chosen
        result = RankResult(
            run_id=msg.run_id,
            item_name=item_name,
            chosen={},
            reason=f"No eligible candidates under ${max_price:.2f}",
            rank_score=0.0,
        )
        await ctx.send(sender, result)
        return

    reason = _build_reason(chosen, item_name, max_price)
    logger.info(f"[ranker] Chose: '{chosen.get('title', '')[:50]}' at ${chosen.get('price', 0):.2f}")

    await post_event(msg.run_id, "ranking_done", {
        "item_name": item_name,
        "chosen_title": chosen.get("title", "")[:80],
        "chosen_price": chosen.get("price", 0),
        "chosen_url": chosen.get("url", ""),
        "reason": reason,
    })

    result = RankResult(
        run_id=msg.run_id,
        item_name=item_name,
        chosen=chosen,
        reason=reason,
        rank_score=0.0,
    )
    await ctx.send(sender, result)
    logger.info(f"[ranker] Sent RankResult for '{item_name}' to orchestrator")


ranker_agent.include(ranker_proto)

if __name__ == "__main__":
    ranker_agent.run()
