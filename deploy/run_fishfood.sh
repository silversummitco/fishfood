#!/usr/bin/env bash
# Launch the Fish Food pond simulation (visual mode) from its own virtualenv.
# Used by the desktop shortcut. Opens a pygame window; no terminal needed.

set -euo pipefail

PROJECT_DIR="/home/arrive/Projects/fishfood"
VENV_PY="$PROJECT_DIR/.venv/bin/python"

cd "$PROJECT_DIR"

if [ ! -x "$VENV_PY" ]; then
  echo "Virtualenv python not found at $VENV_PY" >&2
  echo "Create it with:  python3 -m venv .venv && .venv/bin/pip install numpy pygame" >&2
  exit 1
fi

exec "$VENV_PY" fish_food.py "$@"
