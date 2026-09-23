#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export QWEN_DATA_DIR="${QWEN_DATA_DIR:-$HOME/Library/Application Support/Qwen}"
mkdir -p "$QWEN_DATA_DIR"

if ! command -v python3.12 >/dev/null 2>&1; then
  if ! command -v brew >/dev/null 2>&1; then
    echo 'Install Homebrew from https://brew.sh, then open quickStart.command again.'
    read -r -p 'Press Return to close... ' _
    exit 1
  fi
  brew install python@3.12
fi
if ! command -v node >/dev/null 2>&1 || ! node -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 22 ? 0 : 1)' >/dev/null 2>&1; then
  if ! command -v brew >/dev/null 2>&1; then
    echo 'Install Node.js 22 or newer, then open quickStart.command again.'
    read -r -p 'Press Return to close... ' _
    exit 1
  fi
  brew install node@22
  export PATH="$(brew --prefix node@22)/bin:$PATH"
fi

if [ ! -x .venv/bin/python ]; then
  python3.12 -m venv .venv
fi
if [ ! -f .venv/.qwen-mac-ready ] || [ requirements-mac.txt -nt .venv/.qwen-mac-ready ]; then
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements-mac.txt
  touch .venv/.qwen-mac-ready
fi
if [ ! -d node_modules ]; then npm ci --no-audit --no-fund; fi
if [ ! -f dist/index.html ] || [ -n "$(find src public -type f -newer dist/index.html -print -quit)" ]; then npm run build; fi
if [ ! -f "$QWEN_DATA_DIR/.examples-ready" ]; then
  .venv/bin/python scripts/download_examples.py && touch "$QWEN_DATA_DIR/.examples-ready"
fi

if [ ! -f Qwen.app/Contents/Resources/Qwen.icns ]; then
  .venv/bin/python scripts/create_mac_app.py
fi

echo 'Opening Qwen at http://127.0.0.1:47821'
.venv/bin/python -m uvicorn backend.server:app --host 127.0.0.1 --port 47821 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT INT TERM
for attempt in $(seq 1 40); do
  if curl --silent --fail http://127.0.0.1:47821/api/health >/dev/null; then break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then wait "$server_pid"; exit 1; fi
  sleep 0.5
done
open http://127.0.0.1:47821
wait "$server_pid"
