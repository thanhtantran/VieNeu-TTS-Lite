#!/usr/bin/env bash
# Run VieNeu-TTS v3 Nano OpenAI-compatible API for LAN use on Orange Pi / aarch64.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS_DIR="${VIENEU_MODELS_DIR:-$ROOT/models}"
ONNX_DIR="${VIENEU_ONNX_DIR:-$MODELS_DIR/VieNeu-TTS-v3-Nano}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WEB_ENABLED="${VieNeuTTS_WEB_ENABLED:-true}"
HISTORY_DIR="${VIENEU_WEB_HISTORY_DIR:-$ROOT/models/web-history}"

if [[ ! -d .venv ]]; then
  echo "Missing .venv — run: uv sync" >&2
  exit 1
fi
if [[ ! -f "$ONNX_DIR/config.json" || ! -f "$ONNX_DIR/codec_decoder.onnx" ]]; then
  echo "Missing local Nano model at: $ONNX_DIR" >&2
  echo "Run: uv run python scripts/download_models.py" >&2
  exit 1
fi

mkdir -p "$HISTORY_DIR" "$MODELS_DIR"

export HF_HOME="$MODELS_DIR"
export HUGGINGFACE_HUB_CACHE="$MODELS_DIR/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VIENEU_MODE=v3nano
export VIENEU_ONNX_DIR="$ONNX_DIR"
export VIENEU_WEB_HISTORY_DIR="$HISTORY_DIR"
export VieNeuTTS_WEB_ENABLED="$WEB_ENABLED"
export HOST PORT

# Optional: source local secrets without printing them
if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi

if [[ -z "${VIENEU_API_KEY:-}" && -f "$ROOT/.vieneu_api_key" ]]; then
  VIENEU_API_KEY="$(tr -d '\n' < "$ROOT/.vieneu_api_key")"
  export VIENEU_API_KEY
fi

echo "Starting VieNeu API"
echo "  root:    $ROOT"
echo "  onnx:    $ONNX_DIR"
echo "  listen:  http://$HOST:$PORT"
echo "  web:     VieNeuTTS_WEB_ENABLED=$WEB_ENABLED"
echo "  offline: HF_HUB_OFFLINE=$HF_HUB_OFFLINE"

exec .venv/bin/python -m apps.openai_speech
