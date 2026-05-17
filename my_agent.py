"""Custom forecasting agent using OpenRouter (v3 — with web search).

Routes requests through the OpenAI-compatible API at OpenRouter.
Improvements:
  - Uses OpenRouter's built-in web search plugin for real-time grounding
  - Two-pass approach: search + reason, then commit to probability
  - Category-specific reasoning strategies
  - Stronger calibration nudges to avoid anchoring on 0.50
  - Explicit instruction NOT to hallucinate results

Usage (local module):
    prophet forecast predict --events events.json --local my_agent

The agent returns {"p_yes": float, "rationale": str} for each event,
matching the contract expected by ``prophet forecast predict``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenRouter client (OpenAI-compatible)
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Lazily initialize the OpenAI client pointed at OpenRouter."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to .env or export it directly."
            )
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


# ---------------------------------------------------------------------------
# Model selection — append :online for web search grounding
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "google/gemini-2.5-flash"

# Whether to enable web search. Set FORECAST_WEB_SEARCH=false to disable.
WEB_SEARCH_ENABLED = os.environ.get("FORECAST_WEB_SEARCH", "true").lower() != "false"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a world-class superforecaster. You consistently outperform prediction \
markets and have been trained in the techniques of Philip Tetlock's \
superforecasting methodology.

CRITICAL: You are forecasting the probability that the FIRST listed outcome \
("{first_outcome}") is the correct resolution. This is the "YES" outcome.

YOUR METHODOLOGY — FOLLOW THESE STEPS IN ORDER:

STEP 1 — FIND THE MARKET PRICE:
Search for the CURRENT prediction market odds for this EXACT event on \
Polymarket, Kalshi, Metaculus, or any prediction market. Look for the \
current probability/price for the first outcome ("{first_outcome}").
- If you find market odds, record them as your STARTING BASELINE.
- If you cannot find market odds, use reference class base rates \
(50% for binary, 1/N for multi-outcome).

STEP 2 — GATHER EVIDENCE:
Search for recent news, results, data, and expert analysis about this event.
Extract concrete facts: scores, standings, rankings, poll numbers, \
official results, injury reports, breaking news.

STEP 3 — DECIDE WHETHER TO DEVIATE FROM THE MARKET:
Compare your evidence against the market price. Ask yourself:
- Does my evidence reveal something the market may not have priced in yet?
- Is there breaking news that just happened (within last few hours)?
- Is there a clear, factual reason the market is wrong?
- Has the event already resolved (search confirms the outcome)?

DEVIATION RULES:
- If the event has ALREADY RESOLVED and you found the result: deviate \
strongly (0.90-0.95 if first outcome won, 0.05-0.10 if it lost).
- If you have STRONG evidence the market hasn't priced in: deviate \
by 10-20 percentage points from the market price.
- If you have MODERATE evidence: deviate by 5-10 percentage points.
- If you have WEAK or NO special evidence: STAY CLOSE to the market \
price (within 3-5 percentage points). The market is usually right.
- NEVER deviate more than 25 percentage points from the market unless \
you have confirmed resolution results.

STEP 4 — CALIBRATION CHECK:
Before finalizing, verify your probability is reasonable:
- Is it consistent with the market price given your evidence?
- Are you being overconfident about uncertain information?
- Would you bet real money at these odds?

CRITICAL RULES:
- NEVER fabricate or hallucinate results. If you don't know, stay with \
the market price.
- When in doubt, trust the market. Prediction markets aggregate \
information from thousands of informed participants.
- Your edge comes from SPEED (catching breaking news before markets \
adjust) and RESEARCH DEPTH, not from gut feelings.
- Always report what market price you found (if any) in your rationale.

CONTEXT:
- Current date: {current_date}
- Event close time: {close_time}
- Number of possible outcomes: {n_outcomes}
- First outcome (YES): "{first_outcome}"

{category_guidance}

RESPONSE FORMAT:
Think through your analysis step by step, then provide your final answer as \
valid JSON on the LAST line of your response:
{{"p_yes": <float between 0.05 and 0.95>, "rationale": "<2-4 sentences. \
MUST include the market price you found (if any) and why you did or did \
not deviate from it.>"}}

The JSON must be on its own line at the end. No markdown fences around it."""


# ---------------------------------------------------------------------------
# Category-specific guidance
# ---------------------------------------------------------------------------

CATEGORY_GUIDANCE = {
    "Sports": """\
SPORTS-SPECIFIC GUIDANCE:
- For individual matchups: Consider player/team rankings, recent form, \
head-to-head records, home/away advantage, and surface/venue.
- Home advantage: NBA ~60%, MLB ~54%, Soccer ~46% win + ~27% draw.
- Playoff series: Higher seed wins ~65%. Series leaders rarely blow it.
- League winners: Historical dominance matters (PSG in Ligue 1, \
Real Madrid/Barcelona in La Liga, etc.).
- Cricket: Home advantage ~60%. First-class matches can draw frequently.
- Tennis: Higher-ranked players win ~65-70% in early rounds. \
Smaller tournaments have more upsets.
- If the event has already occurred and you find the result via search, \
report it confidently.""",

    "Entertainment": """\
ENTERTAINMENT-SPECIFIC GUIDANCE:
- Reality TV: inherently unpredictable. Base rate = ~1/N to 2/N.
- For competition winners with many contestants, start with 1/N and only \
adjust with strong evidence (e.g. search results showing the winner).
- Celebrity events: Consider cultural relevance and upcoming projects.
- If the show has already aired and you find the result, be confident.""",

    "Elections": """\
ELECTIONS-SPECIFIC GUIDANCE:
- Incumbency advantage: sitting officials win ~80-90% of primaries.
- No incumbent: look for name recognition, endorsements, fundraising.
- General elections: Consider partisan lean of district/state.
- If you know nothing specific, use structural factors rather than 0.50.
- Search for recent polling data or endorsements.""",

    "Politics": """\
POLITICS-SPECIFIC GUIDANCE:
- Incumbent leaders: strong incumbency advantage (~70-80% retention).
- Policy outcomes: Consider political alignment of decision-makers.
- Legislative votes: Party-line voting is common in polarized environments.
- International politics: Status quo bias is strong — predict continuity.
- Search for recent news about political developments.""",

    "Economics": """\
ECONOMICS-SPECIFIC GUIDANCE:
- Economic indicators tend to follow trends. Mean reversion is real but slow.
- Central bank decisions: usually well-telegraphed via forward guidance.
- GDP/inflation: consensus survey forecasts are decent base rates.
- Market expectations (futures, swaps) are strong priors.""",
}


def _get_category_guidance(category: str) -> str:
    """Get category-specific prompt guidance."""
    return CATEGORY_GUIDANCE.get(category, "No specific category guidance available.")


def _build_user_prompt(event: dict) -> str:
    """Construct the user-facing prompt from event fields."""
    parts = [f"EVENT: {event.get('title', 'Unknown event')}"]

    if event.get("subtitle"):
        parts.append(f"Subtitle: {event['subtitle']}")

    if event.get("description"):
        parts.append(f"Description: {event['description']}")

    if event.get("rules"):
        parts.append(f"Resolution rules: {event['rules']}")

    parts.append(f"Category: {event.get('category', 'Unknown')}")
    parts.append(f"Close time: {event.get('close_time', 'Unknown')}")

    # Include market ticker to help the model find this exact market
    if event.get("market_ticker"):
        parts.append(f"Market ticker: {event['market_ticker']}")

    outcomes = event.get("outcomes", [])
    if outcomes:
        parts.append(f"\nPossible outcomes ({len(outcomes)} total):")
        for i, o in enumerate(outcomes):
            marker = " ← THIS IS THE 'YES' OUTCOME" if i == 0 else ""
            parts.append(f"  {i+1}. {o}{marker}")

        parts.append(
            f"\nYou must estimate the probability that \"{outcomes[0]}\" "
            f"is the correct/winning outcome."
        )

        if len(outcomes) == 2:
            parts.append(
                f"This is a binary market. If \"{outcomes[0]}\" doesn't win, "
                f"then \"{outcomes[1]}\" wins."
            )
        else:
            parts.append(
                f"This is a multi-outcome market with {len(outcomes)} options. "
                f"The base rate for any single outcome is ~{1/len(outcomes):.1%}. "
                f"Only adjust from this base rate if you have specific evidence."
            )

    parts.append(
        "\nINSTRUCTIONS:"
        "\n1. FIRST: Search for current prediction market odds for this exact "
        "event on Polymarket, Kalshi, or similar platforms."
        "\n2. THEN: Search for recent news and evidence about this event."
        "\n3. Use the market price as your baseline. Only deviate if you have "
        "strong evidence the market hasn't priced in."
        "\n4. Report the market price you found in your rationale."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core forecasting logic
# ---------------------------------------------------------------------------


def _parse_llm_response(text: str) -> dict:
    """Parse the LLM's JSON response, trying multiple strategies.

    The model may return the JSON in various formats:
      - On its own line at the end
      - Wrapped in markdown code fences
      - With markdown links in the rationale (common with web search)
      - Spread across multiple lines
    """
    text = text.strip()

    # Strategy 1: Find JSON on the last non-empty line
    lines = text.split("\n")
    for line in reversed(lines):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        if line.startswith("{") and "p_yes" in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    # Strategy 2: Greedy regex that handles rationale with special chars
    # This captures from {"p_yes" to the last } on the same or following lines
    json_match = re.search(
        r'\{\s*"p_yes"\s*:\s*([\d.]+)\s*,\s*"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
        text,
        re.DOTALL,
    )
    if json_match:
        try:
            p_val = float(json_match.group(1))
            rationale = json_match.group(2)
            # Unescape common escape sequences
            rationale = rationale.replace('\\"', '"').replace("\\n", " ")
            return {"p_yes": p_val, "rationale": rationale}
        except (ValueError, IndexError):
            pass

    # Strategy 3: Try to find any JSON block containing p_yes
    # Use a balanced-brace approach for multi-line JSON
    for match in re.finditer(r'\{', text):
        start = match.start()
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    if "p_yes" in candidate:
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            break
                    break

    # Strategy 4: Strip markdown fences and try parsing
    clean = text
    # Remove ```json or ``` prefix
    clean = re.sub(r'```(?:json)?\s*\n?', '', clean)
    clean = clean.strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Strategy 5: Extract just the p_yes value via simple regex
    # Last resort — at least get the probability
    p_match = re.search(r'"p_yes"\s*:\s*([\d.]+)', text)
    if p_match:
        p_val = float(p_match.group(1))
        # Try to extract rationale too
        r_match = re.search(r'"rationale"\s*:\s*"([^"]*)"', text)
        rationale = r_match.group(1) if r_match else "Parsed from partial response"
        return {"p_yes": p_val, "rationale": rationale}

    raise ValueError(f"Could not parse LLM response: {text[:200]}...")


def forecast(event: dict) -> dict:
    """Call the LLM via OpenRouter to produce a probability forecast.

    Uses web search plugin for real-time grounding when available.

    Args:
        event: Event dict with keys like market_ticker, title, etc.

    Returns:
        Dict with "p_yes" (float) and "rationale" (str).
    """
    client = _get_client()
    base_model = os.environ.get("FORECAST_MODEL", DEFAULT_MODEL)

    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    close_time = event.get("close_time", "Unknown")
    outcomes = event.get("outcomes", [])
    n_outcomes = len(outcomes)
    first_outcome = outcomes[0] if outcomes else "Unknown"
    category = event.get("category", "Unknown")

    system_prompt = SYSTEM_PROMPT.format(
        current_date=current_date,
        close_time=close_time,
        n_outcomes=n_outcomes,
        first_outcome=first_outcome,
        category_guidance=_get_category_guidance(category),
    )

    user_prompt = _build_user_prompt(event)

    # Build request kwargs — add web search plugin if enabled
    extra_body = {}
    model = base_model
    if WEB_SEARCH_ENABLED:
        # Use the :online suffix for web search grounding
        model = f"{base_model}:online"

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = response.choices[0].message.content
        if not text:
            raise ValueError("Empty response from LLM")

        try:
            data = _parse_llm_response(text)
        except (ValueError, json.JSONDecodeError):
            # Retry: ask the model to just give us the JSON based on its analysis
            logger.info(
                "Parse failed for %s — retrying with JSON-only prompt",
                event.get("market_ticker", "?"),
            )
            retry_response = client.chat.completions.create(
                model=base_model,  # No :online — we already have the analysis
                max_tokens=200,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "Extract the probability estimate from the analysis below. Respond with ONLY valid JSON: {\"p_yes\": <float>, \"rationale\": \"<summary>\"}"},
                    {"role": "user", "content": f"Based on this analysis, provide the JSON:\n\n{text[:2000]}"},
                ],
            )
            retry_text = retry_response.choices[0].message.content or ""
            data = _parse_llm_response(retry_text)

        p = max(0.05, min(0.95, float(data["p_yes"])))
        rationale = data.get("rationale", "")

        logger.info(
            "Forecast for %s: p_yes=%.3f — %s",
            event.get("market_ticker", "?"),
            p,
            rationale[:80],
        )

        return {"p_yes": p, "rationale": rationale}

    except Exception as e:
        logger.warning(
            "LLM call failed for %s: %s — using base rate fallback",
            event.get("market_ticker", "?"),
            e,
        )
        # Smarter fallback: use base rate instead of flat 0.50
        if n_outcomes > 2:
            fallback_p = max(0.05, min(0.95, 1.0 / n_outcomes))
        else:
            fallback_p = 0.50
        return {
            "p_yes": fallback_p,
            "rationale": f"Fallback to base rate ({fallback_p:.2f}) due to error: {e}",
        }


# ---------------------------------------------------------------------------
# Entry point for `prophet forecast predict --local my_agent`
# ---------------------------------------------------------------------------


def predict(event: dict) -> dict:
    """Predict function for --local mode.

    Args:
        event: Event dict with keys like market_ticker, title, category, etc.

    Returns:
        Dict with p_yes (float) and rationale (str).
    """
    return forecast(event)
