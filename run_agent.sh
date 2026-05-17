#!/usr/bin/env bash
# ============================================================================
# run_agent.sh — Start the Prophet Arena Forecasting Agent
#
# This script is provided for organizers to run the agent in a standardized
# environment. It installs dependencies, validates the environment, and starts
# the FastAPI server on port 8000.
#
# Prerequisites:
#   - Python 3.11+
#   - An OpenRouter API key set as OPENAI_API_KEY
#
# Usage:
#   export OPENAI_API_KEY="sk-or-v1-..."
#   export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
#   bash run_agent.sh
#
# Or use Docker:
#   docker build -t prophet-agent .
#   docker run -p 8000:8000 -e OPENAI_API_KEY="sk-or-v1-..." \
#     -e OPENAI_BASE_URL="https://openrouter.ai/api/v1" prophet-agent
# ============================================================================

set -euo pipefail

echo "========================================="
echo "  Prophet Arena Forecasting Agent"
echo "  Team: Future Unemployed"
echo "========================================="

# Check Python version
PYTHON=${PYTHON:-python3}
echo "[1/4] Checking Python version..."
$PYTHON --version

# Check for API key
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo ""
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "Please export your OpenRouter API key:"
    echo "  export OPENAI_API_KEY=\"sk-or-v1-...\""
    echo "  export OPENAI_BASE_URL=\"https://openrouter.ai/api/v1\""
    exit 1
fi

echo "[2/4] OPENAI_API_KEY is set ✓"

# Install dependencies
echo "[3/4] Installing dependencies..."
$PYTHON -m pip install -q -r requirements.txt

# Start the server
echo "[4/4] Starting FastAPI server on port ${FORECAST_PORT:-8000}..."
echo ""
echo "Agent will be available at:"
echo "  POST http://localhost:${FORECAST_PORT:-8000}/predict"
echo "  GET  http://localhost:${FORECAST_PORT:-8000}/health"
echo ""

$PYTHON server.py
