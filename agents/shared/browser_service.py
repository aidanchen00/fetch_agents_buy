"""
Browser automation service using Browser Use Cloud SDK v3.

Browser Use sends a natural-language task to an AI browser agent
running in the cloud.  The agent returns structured results.

Two high-level operations are exposed:
  - search_amazon(query, …) — search Amazon and return product candidates
  - buy_product(url, quantity, …) — navigate to a product page and add to cart

Usage:
    from agents.shared.browser_service import get_browser_use_service
    svc = get_browser_use_service()
    session_id, live_url, candidates = await svc.search_amazon("AA batteries")
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agents.shared.config import BROWSER_USE_API_KEY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schemas (sent to Browser Use as output_schema)
# ---------------------------------------------------------------------------

class ProductCandidate(BaseModel):
    title: str
    price: float
    rating: Optional[float] = None
    review_count: Optional[int] = None
    url: str
    asin: Optional[str] = None

class SearchResultsOutput(BaseModel):
    products: List[ProductCandidate] = Field(default_factory=list)

class AddToCartOutput(BaseModel):
    success: bool = False
    final_price: Optional[float] = None
    error: Optional[str] = None
    confirmation_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Lightweight result wrappers returned to callers
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    session_id: str
    live_url: str
    candidates: List[Dict[str, Any]]

@dataclass
class BuyProductResult:
    session_id: str
    live_url: str
    success: bool
    final_price: float
    error: str
    recording_url: str


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class BrowserUseService:
    """Wraps Browser Use Cloud SDK v3."""

    def __init__(self):
        from browser_use_sdk.v3 import AsyncBrowserUse  # type: ignore
        self._client = AsyncBrowserUse(api_key=BROWSER_USE_API_KEY)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _poll_session(self, session_id: str, timeout: float = 300):
        """Poll until a session reaches a terminal status."""
        start = time.time()
        while time.time() - start < timeout:
            session = await self._client.sessions.get(session_id)
            if session.status in ("completed", "failed", "stopped"):
                return session
            await asyncio.sleep(2)
        raise TimeoutError(f"Browser Use session {session_id} timed out after {timeout}s")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_amazon(
        self,
        query: str,
        max_results: int = 8,
    ) -> SearchResult:
        """
        Ask Browser Use to search Amazon and extract product candidates.
        Returns a SearchResult with session_id, live_url, and candidate dicts.
        """
        task = (
            f"Go to https://www.amazon.com and search for '{query}'. "
            f"From the search results page, extract the top {max_results} products that have a visible price. "
            "For each product extract: title, price (numeric, USD), star rating (float, if shown), "
            "number of reviews (integer, if shown), product URL (full https://www.amazon.com/dp/ASIN link), "
            "and ASIN (the 10-character alphanumeric ID from the URL). "
            "Skip sponsored results if possible. Only include products with a price > 0."
        )

        logger.info(f"[browser-use] Creating search session for: '{query}'")
        session = await self._client.sessions.create(
            task=task,
            output_schema=SearchResultsOutput,
        )

        session_id = session.id
        live_url = getattr(session, "live_url", "") or ""
        logger.info(f"[browser-use] Search session {session_id} live_url={live_url[:80]}")

        result_session = await self._poll_session(session_id, timeout=120)

        candidates: List[Dict[str, Any]] = []
        output = getattr(result_session, "output", None)
        if output and hasattr(output, "products"):
            candidates = [p.model_dump() for p in output.products]
        elif isinstance(output, dict) and "products" in output:
            candidates = output["products"]
        elif isinstance(output, SearchResultsOutput):
            candidates = [p.model_dump() for p in output.products]

        logger.info(f"[browser-use] Search returned {len(candidates)} candidates")
        return SearchResult(session_id=session_id, live_url=live_url, candidates=candidates[:max_results])

    # ------------------------------------------------------------------
    # Buy / add-to-cart
    # ------------------------------------------------------------------

    async def buy_product(
        self,
        product_url: str,
        quantity: int = 1,
        item_name: str = "",
        max_price: float = 9999,
    ) -> BuyProductResult:
        """
        Ask Browser Use to navigate to *product_url* on Amazon,
        set the quantity, and click Add to Cart.
        """
        qty_instruction = f"Set the quantity to {quantity}. " if quantity > 1 else ""
        task = (
            f"Go to {product_url} on Amazon.com. "
            f"This is a product page for '{item_name}'. "
            f"{qty_instruction}"
            "Click the 'Add to Cart' button. "
            "Wait for the confirmation that the item has been added to cart. "
            "Report whether the add-to-cart was successful, the final price shown on the page, "
            "and any error or confirmation text."
        )

        logger.info(f"[browser-use] Creating buy session for '{item_name}' url={product_url[:60]}")
        session = await self._client.sessions.create(
            task=task,
            output_schema=AddToCartOutput,
        )

        session_id = session.id
        live_url = getattr(session, "live_url", "") or ""
        logger.info(f"[browser-use] Buy session {session_id} live_url={live_url[:80]}")

        result_session = await self._poll_session(session_id, timeout=180)

        success = False
        final_price = 0.0
        error = ""
        output = getattr(result_session, "output", None)
        if output and hasattr(output, "success"):
            success = output.success
            final_price = float(output.final_price or 0)
            error = output.error or ""
        elif isinstance(output, dict):
            success = output.get("success", False)
            final_price = float(output.get("final_price") or 0)
            error = output.get("error", "")

        # Try to get recording URL
        recording_url = ""
        try:
            urls = await self._client.sessions.wait_for_recording(session_id)
            if urls:
                recording_url = urls[0] if isinstance(urls[0], str) else str(urls[0])
        except Exception as e:
            logger.debug(f"[browser-use] No recording for {session_id}: {e}")

        logger.info(f"[browser-use] Buy result for '{item_name}': success={success} price={final_price}")
        return BuyProductResult(
            session_id=session_id,
            live_url=live_url,
            success=success,
            final_price=final_price,
            error=error,
            recording_url=recording_url,
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_service: Optional[BrowserUseService] = None


def get_browser_use_service() -> BrowserUseService:
    """Return a shared BrowserUseService instance."""
    global _service
    if _service is None:
        _service = BrowserUseService()
    return _service
