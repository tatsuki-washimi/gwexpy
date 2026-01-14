#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if ! command -v xvfb-run >/dev/null 2>&1; then
  echo "ERROR: xvfb-run not found. Install Xvfb/xvfb-run or run without headless mode." >&2
  exit 127
fi

echo "Running GUI Tests using pytest-qt and xvfb..."
# -o log_cli=true enables visibility of our logger.info calls
# -v gives verbose test names
xvfb-run -a -s "-screen 0 1920x1080x24" \
  env PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -u -m pytest -p pytestqt.plugin \
  -o log_cli=true -o log_cli_level=INFO \
  ${PYTEST_ARGS:--v} tests/gui/integration
