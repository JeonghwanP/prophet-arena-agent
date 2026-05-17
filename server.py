"""Prophet Arena Forecasting Agent — FastAPI HTTP Server.

Wraps the core forecasting logic from ``my_agent.py`` in a FastAPI server
that exposes the ``POST /predict`` endpoint required by the organizers.

Includes a **Smart Cache** to keep API costs under the $50 OpenRouter budget
during the 2-week continuous evaluation window:
  - Events closing > 24h away → cache for 12 hours
  - Events closing within 24h  → cache for 1 hour
  - Events already past close  → cache for 30 minutes (result might just
    have appeared in search)

Usage:
    python server.py                    # starts on 0.0.0.0:8000
    FORECAST_PORT=9000 python server.py # custom port
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Import the core forecasting engine
import my_agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Prophet Arena Forecasting Agent",
    description="Superforecaster agent powered by Gemini 2.5 Flash + Web Search",
    version="3.0",
)


# ---------------------------------------------------------------------------
# Request / Response models (match the CLI's predict contract)
# ---------------------------------------------------------------------------

class EventRequest(BaseModel):
    """Incoming event payload from the organizers' evaluation harness."""
    event_ticker: str
    market_ticker: str
    title: str
    subtitle: str | None = None
    description: str | None = None
    category: str
    rules: str | None = None
    close_time: str
    outcomes: list[str] = []

class OutcomeProbability(BaseModel):
    market: str
    p_yes: float

class PredictionResponse(BaseModel):
    """Response returned to the evaluation harness."""
    probabilities: list[OutcomeProbability]
    rationale: str


# ---------------------------------------------------------------------------
# Smart Cache
# ---------------------------------------------------------------------------

class SmartCache:
    """Time-aware prediction cache that adapts TTL based on event urgency.

    Cache TTL logic:
      - Event closes in > 24 hours → cache for 12 hours
      - Event closes in ≤ 24 hours → cache for 1 hour
      - Event already closed        → cache for 30 minutes

    This balances cost savings (avoid redundant API calls) with freshness
    (catch newly resolved events quickly).
    """

    # TTLs in seconds
    TTL_FAR = 12 * 3600      # 12 hours for far-future events
    TTL_SOON = 1 * 3600      # 1 hour for events closing within 24h
    TTL_PAST = 30 * 60       # 30 minutes for events that already closed

    def __init__(self):
        # key: market_ticker → (prediction_dict, timestamp, ttl_seconds)
        self._store: dict[str, tuple[dict, float, float]] = {}
        self._hits = 0
        self._misses = 0

    def _compute_ttl(self, close_time_str: str) -> float:
        """Determine cache TTL based on how far away the close time is."""
        try:
            close_time = datetime.fromisoformat(
                close_time_str.replace("Z", "+00:00")
            )
            now = datetime.now(timezone.utc)
            hours_until_close = (close_time - now).total_seconds() / 3600

            if hours_until_close <= 0:
                # Event already closed — result may have just appeared
                return self.TTL_PAST
            elif hours_until_close <= 24:
                # Closing soon — check more frequently
                return self.TTL_SOON
            else:
                # Far future — safe to cache longer
                return self.TTL_FAR
        except (ValueError, TypeError):
            # Can't parse close_time — default to 1 hour
            return self.TTL_SOON

    def get(self, market_ticker: str) -> dict | None:
        """Return cached prediction if still valid, else None."""
        entry = self._store.get(market_ticker)
        if entry is None:
            self._misses += 1
            return None

        prediction, cached_at, ttl = entry
        age = time.time() - cached_at

        if age > ttl:
            # Expired
            del self._store[market_ticker]
            self._misses += 1
            logger.info(
                "Cache EXPIRED for %s (age=%.0fs, ttl=%.0fs)",
                market_ticker, age, ttl,
            )
            return None

        self._hits += 1
        logger.info(
            "Cache HIT for %s (age=%.0fs, ttl=%.0fs, remaining=%.0fs)",
            market_ticker, age, ttl, ttl - age,
        )
        return prediction

    def put(self, market_ticker: str, close_time: str, prediction: dict):
        """Store a prediction with a smart TTL."""
        ttl = self._compute_ttl(close_time)
        self._store[market_ticker] = (prediction, time.time(), ttl)
        logger.info(
            "Cache STORE for %s (ttl=%.0fs / %.1fh)",
            market_ticker, ttl, ttl / 3600,
        )

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "cached_entries": len(self._store),
            "total_lookups": total,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 1),
        }


# Global cache instance
_cache = SmartCache()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(event: EventRequest) -> PredictionResponse:
    """Receive an event and return a probability forecast.

    Uses the smart cache to avoid redundant API calls while keeping
    predictions fresh for time-sensitive events.
    """
    ticker = event.market_ticker
    logger.info("Request: %s — %s", ticker, event.title[:60])

    # Check cache first
    cached = _cache.get(ticker)
    if cached is not None:
        # The cache stores our internal format {"p_yes": float, "rationale": str}
        # We need to convert it to the array format for the response.
        p_yes = cached["p_yes"]
        rationale = cached["rationale"]
    else:
        # Cache miss — run the forecasting engine
        event_dict = event.model_dump()
        result = my_agent.forecast(event_dict)

        # Store in cache with smart TTL (store internal format)
        _cache.put(ticker, event.close_time, result)
        
        p_yes = result["p_yes"]
        rationale = result["rationale"]

    # Convert our internal `p_yes` (for the FIRST outcome) into a `probabilities` array
    n_outcomes = max(1, len(event.outcomes))
    
    if n_outcomes <= 1:
        probs = [p_yes]
    else:
        # p_yes is the probability for outcomes[0].
        # Distribute the remaining probability equally among the remaining outcomes.
        p_other = (1.0 - p_yes) / (n_outcomes - 1)
        probs = [p_yes] + [p_other] * (n_outcomes - 1)
        
        # Make sure they exactly sum to 1.0 (float math precision)
        probs = [round(p, 4) for p in probs]

    # Build the required array of objects
    prob_objects = []
    if event.outcomes:
        for i, outcome_name in enumerate(event.outcomes):
            prob_objects.append(OutcomeProbability(market=outcome_name, p_yes=probs[i]))
    else:
        # Fallback if no outcomes provided
        prob_objects.append(OutcomeProbability(market="YES", p_yes=probs[0]))

    return PredictionResponse(probabilities=prob_objects, rationale=rationale)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "agent_version": "3.0",
        "model": os.environ.get("FORECAST_MODEL", my_agent.DEFAULT_MODEL),
        "web_search": my_agent.WEB_SEARCH_ENABLED,
        "cache": _cache.stats,
    }


@app.get("/")
async def root():
    """Root endpoint — basic info."""
    return {
        "name": "Prophet Arena Forecasting Agent",
        "version": "3.0",
        "predict_endpoint": "/predict",
        "health_endpoint": "/health",
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    port = int(os.environ.get("FORECAST_PORT", "8000"))
    logger.info("Starting Prophet Arena agent on port %d", port)
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
