#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "ERROR: xvfb-run not found. Install Xvfb/xvfb-run or run without headless mode." >&2
  exit 127
fi

xvfb-run -a -s "-screen 0 1920x1080x24" \
  env PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -u -m pytest -p pytestqt.plugin ${PYTEST_ARGS:--q} tests/gui tests/e2e
