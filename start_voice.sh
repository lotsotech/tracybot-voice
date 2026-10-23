#!/usr/bin/env bash
# Start the TracyBot Always-On Voice Assistant
# Usage: ./start_voice.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/voice_venv/bin/python"

if [[ ! -f "$VENV_PYTHON" ]]; then
  echo "ERROR: voice_venv not found at $SCRIPT_DIR/voice_venv"
  echo "Run: cd $SCRIPT_DIR && python3 -m venv voice_venv && voice_venv/bin/pip install fastapi uvicorn[standard] piper-tts openai-whisper httpx"
  exit 1
fi

echo "Starting TracyBot Always-On Voice..."
exec "$VENV_PYTHON" "$SCRIPT_DIR/always_on_voice.py"
