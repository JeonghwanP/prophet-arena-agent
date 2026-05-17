# Future Unemployed - AI Prophet Forecasting Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A high-performance forecasting agent built for the **Prophet Arena Hackathon (Forecasting Track)**. 

The agent uses **Gemini 2.5 Flash** connected to **OpenRouter's Web Search** plugin to perform real-time research. It employs a "market-anchoring" strategy: it first searches for live odds on Polymarket/Kalshi and uses them as a baseline, only deviating when it uncovers compelling, unpriced evidence.

To protect API budgets during the continuous 2-week evaluation window, the agent is wrapped in a **FastAPI server with an Adaptive Smart Cache**.

## Project Structure
- `my_agent.py` — Core forecasting logic, web search integration, and robust JSON parsing/retries.
- `server.py` — FastAPI HTTP server implementing the `/predict` endpoint and the Smart Cache.
- `run_agent.sh` — Organizer-friendly script to launch the agent.
- `Dockerfile` — Docker configuration for easy cloud deployment.

## Running the Agent (For Organizers)

We have provided a unified script that checks dependencies, installs them, and starts the server on port 8000.

**Prerequisites:** Python 3.11+ and an OpenRouter API key.

```bash
# 1. Export your API credentials
export OPENAI_API_KEY="sk-or-v1-..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"

# 2. Run the agent script
bash run_agent.sh
```

The agent will be available at:
- Prediction endpoint: `POST http://localhost:8000/predict`
- Health/Stats endpoint: `GET http://localhost:8000/health`

### Using Docker
If you prefer running via Docker:
```bash
docker build -t prophet-agent .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="sk-or-v1-..." \
  -e OPENAI_BASE_URL="https://openrouter.ai/api/v1" \
  prophet-agent
```

## Architecture Details

- **Model:** `google/gemini-2.5-flash:online` (using Exa web search).
- **Caching:** 
  - Events closing >24 hours away: Cached for 12 hours.
  - Events closing <24 hours away: Cached for 1 hour.
  - Already closed events: Cached for 30 minutes.
- **Resilience:** Features a 5-strategy regex JSON parser. If parsing fails (e.g., model outputs markdown links instead of JSON), the agent automatically runs a cheap, text-only retry prompt to extract the probability.

## License
MIT. See `LICENSE`.
