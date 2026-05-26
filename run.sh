#!/usr/bin/env bash
# Thin POSIX wrapper around `python -m sd_compress`.
#
# Works on Linux (recommended) and macOS. For Windows use `run.ps1`.
#
# Usage:
#   ./run.sh                    # run the full pipeline + launch server
#   ./run.sh --no-serve         # full pipeline without the Gradio server
#   ./run.sh distill-progressive  # run a single stage
#   ./run.sh evaluate --stage distilled --model-dir ./output/distilled
#
# Configuration is taken from the environment (see sd_compress/config.py). The
# defaults match what the original monolithic script used.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

###############################################################################
# Virtual environment
###############################################################################
if [ ! -d "$VENV_DIR" ]; then
    echo "[run.sh] Creating virtual environment in $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip >/dev/null

###############################################################################
# Dependencies
###############################################################################
if [ ! -f "$VENV_DIR/.deps_installed" ]; then
    echo "[run.sh] Installing Python requirements (this may take a while)"
    pip install -r requirements.txt

    # CLIP for evaluation metrics is hosted on GitHub
    pip install --quiet "git+https://github.com/openai/CLIP.git" \
        || echo "[run.sh] WARNING: CLIP install failed; CLIP score will be skipped"

    # xFormers is best-effort; failures are non-fatal
    pip install --quiet xformers \
        || echo "[run.sh] NOTE: xformers unavailable on this platform/CUDA combination"

    touch "$VENV_DIR/.deps_installed"
fi

###############################################################################
# Pipeline
###############################################################################
if [ $# -eq 0 ]; then
    # Default: full pipeline followed by the Gradio server, matching legacy behaviour.
    exec python -m sd_compress run --serve
fi

# Support `./run.sh --no-serve` as a shortcut for the full pipeline without the
# UI; otherwise forward arguments verbatim to the CLI.
if [ "$1" = "--no-serve" ]; then
    exec python -m sd_compress run
fi

exec python -m sd_compress "$@"
