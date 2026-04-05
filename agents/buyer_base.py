"""
Shared buyer agent logic. Each buyer_[a-e]/agent.py imports and uses this.

Buyer agent responsibilities:
1. Receive BuyRequest from orchestrator
2. Use Browser Use Cloud to navigate to the product page and add to cart
3. Return BuyResultMsg to orchestrator
4. Post all events/session metadata to FastAPI
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from uagents import Agent, Context, Protocol

from agents.shared.browser_service import get_browser_use_service
from agents.shared.config import FASTAPI_CALLBACK_URL
from agents.shared.messages import BuyRequest, BuyResultMsg

logger = logging.getLogger(__name__)


async def post_event(run_id: str, agent_name: str, event_type: str, payload: dict):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(FASTAPI_CALLBACK_URL, json={
                "run_id": run_id,
                "agent_name": agent_name,
                "event_type": event_type,
                "timestamp": datetime.utcnow().isoformat(),
                "payload": payload,
            })
    except Exception as e:
        logger.warning(f"[{agent_name}] Failed to post event {event_type}: {e}")


async def execute_buy(
    ctx: Context,
    sender: str,
    msg: BuyRequest,
    agent_name: str,
):
    """Core buy logic shared across all buyer agents."""
    item_name = msg.item.get("name", "unknown")
    max_price = float(msg.item.get("max_price", 9999))
    quantity = msg.quantity
    product = msg.chosen_product
    product_url = product.get("url", "")

    logger.info(f"[{agent_name}] Buy request: '{item_name}' qty={quantity} url={product_url[:60]}")

    await post_event(msg.run_id, agent_name, "buy_started", {
        "item_name": item_name,
        "product_url": product_url,
        "quantity": quantity,
        "approval_ref": msg.approval_ref,
    })

    result_dict = {
        "run_id": msg.run_id,
        "item_name": item_name,
        "status": "failed",
        "final_price": 0.0,
        "quantity": quantity,
        "screenshot_url": "",
        "screenshot_path": "",
        "live_view_url": "",
        "session_id": "",
        "agent_name": agent_name,
        "error": "",
    }

    if not product_url:
        result_dict["error"] = "No product URL in chosen product"
    else:
        async def _on_session(sid: str, url: str):
            result_dict["session_id"] = sid
            result_dict["live_view_url"] = url
            await post_event(msg.run_id, agent_name, "session_created", {
                "session_id": sid,
                "live_view_url": url,
                "debugger_url": "",
                "item_name": item_name,
            })

        try:
            svc = get_browser_use_service()
            buy_result = await svc.buy_product(
                product_url=product_url,
                quantity=quantity,
                item_name=item_name,
                max_price=max_price,
                on_session_created=_on_session,
            )

            result_dict["session_id"] = buy_result.session_id
            result_dict["live_view_url"] = buy_result.live_url

            if buy_result.success:
                result_dict["status"] = "success"
                result_dict["final_price"] = buy_result.final_price or float(product.get("price", 0))
                logger.info(f"[{agent_name}] Successfully added '{item_name}' to cart")
            else:
                result_dict["status"] = "failed"
                result_dict["error"] = buy_result.error or "Add-to-cart failed"
                logger.error(f"[{agent_name}] Add to cart failed for '{item_name}': {buy_result.error}")

            if buy_result.recording_url:
                result_dict["screenshot_url"] = buy_result.recording_url
                result_dict["screenshot_path"] = buy_result.recording_url

        except Exception as e:
            logger.error(f"[{agent_name}] Buy execution error for '{item_name}': {e}")
            result_dict["error"] = str(e)

    await post_event(msg.run_id, agent_name, "buy_done", {
        "item_name": item_name,
        "status": result_dict["status"],
        "screenshot_url": result_dict["screenshot_url"],
        "live_view_url": result_dict["live_view_url"],
        "session_id": result_dict["session_id"],
        "error": result_dict.get("error", ""),
    })

    if result_dict.get("screenshot_url"):
        await post_event(msg.run_id, agent_name, "screenshot_saved", {
            "item_name": item_name,
            "screenshot_url": result_dict["screenshot_url"],
            "screenshot_path": result_dict["screenshot_path"],
        })

    response = BuyResultMsg(**result_dict)
    await ctx.send(sender, response)
    logger.info(f"[{agent_name}] Sent BuyResultMsg for '{item_name}' (status={result_dict['status']})")


def make_buyer_agent(name: str, seed: str, port: int) -> tuple[Agent, Protocol]:
    """Factory: creates a buyer agent + protocol."""
    agent = Agent(
        name=name,
        seed=seed,
        port=port,
        mailbox=True,
    )
    proto = Protocol(name=f"{name}-protocol")

    @proto.on_message(BuyRequest)
    async def handle_buy(ctx: Context, sender: str, msg: BuyRequest):
        await execute_buy(ctx, sender, msg, agent_name=name)

    agent.include(proto)
    return agent, proto
