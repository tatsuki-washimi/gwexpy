#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

mode="${1:-pytest}"
shift || true

target="${GUI_TEST_TARGET:-tests/gui/integration}"

pytest_args=()
if [ -n "${PYTEST_ARGS:-}" ]; then
  read -r -a pytest_args <<< "${PYTEST_ARGS}"
else
  pytest_args=(-v)
fi
if [ "$#" -gt 0 ]; then
  pytest_args+=("$@")
fi
has_markexpr=0
for arg in "${pytest_args[@]}"; do
  case "$arg" in
    -m|--markexpr)
      has_markexpr=1
      break
      ;;
    -m*|--markexpr=*)
      has_markexpr=1
      break
      ;;
  esac
done
if [ "$has_markexpr" -eq 0 ]; then
  pytest_args+=(-m "not nds")
fi

export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

read -r has_pytest_xvfb pytest_xvfb_arg <<< "$(python - <<'PY'
import importlib.util
import inspect

if not importlib.util.find_spec("pytest_xvfb"):
    print("0")
    raise SystemExit(0)

import pytest_xvfb

arg = ""
if hasattr(pytest_xvfb, "pytest_addoption"):
    try:
        src = inspect.getsource(pytest_xvfb.pytest_addoption)
    except OSError:
        src = ""
    if 'addoption("--xvfb"' in src or "addoption('--xvfb'" in src:
        arg = "--xvfb"

print("1", arg)
PY
)"

pytest_xvfb_args=()
if [ "$has_pytest_xvfb" -eq 1 ] && [ -n "$pytest_xvfb_arg" ]; then
  pytest_xvfb_args=("$pytest_xvfb_arg")
fi

has_xvfb_run=0
if command -v xvfb-run >/dev/null 2>&1; then
  has_xvfb_run=1
fi

xvfb_usable=1
if [ ! -d /tmp/.X11-unix ] || [ ! -k /tmp/.X11-unix ]; then
  xvfb_usable=0
fi

if [ "$xvfb_usable" -eq 0 ]; then
  echo "WARN: /tmp/.X11-unix missing sticky bit; disabling Xvfb support." >&2
  has_xvfb_run=0
  has_pytest_xvfb=0
fi

use_pytest_xvfb=0
if [ "${GUI_USE_PYTEST_XVFB:-0}" = "1" ]; then
  if [ "$has_pytest_xvfb" -eq 1 ]; then
    use_pytest_xvfb=1
  fi
elif [ "$has_xvfb_run" -eq 1 ]; then
  use_pytest_xvfb=0
elif [ "$has_pytest_xvfb" -eq 1 ]; then
  use_pytest_xvfb=1
fi

use_xvfb=0
if [ "$use_pytest_xvfb" -eq 1 ] || [ "$has_xvfb_run" -eq 1 ]; then
  use_xvfb=1
fi

if [ -z "${QT_QPA_PLATFORM:-}" ]; then
  if [ "$use_xvfb" -eq 1 ]; then
    export QT_QPA_PLATFORM="xcb"
  else
    export QT_QPA_PLATFORM="minimal"
  fi
fi

pytest_cmd=(python -u -m pytest -p pytestqt.plugin -o log_cli=true -o log_cli_level=INFO)
if [ "$use_pytest_xvfb" -eq 1 ]; then
  pytest_cmd+=(-p pytest_xvfb "${pytest_xvfb_args[@]}")
fi
pytest_cmd+=("${pytest_args[@]}" "$target")

case "$mode" in
  pytest)
    runner_cmd=("${pytest_cmd[@]}")
    ;;
  gdb)
    if ! command -v gdb >/dev/null 2>&1; then
      echo "ERROR: gdb not found." >&2
      exit 127
    fi
    ulimit -c unlimited || true
    runner_cmd=(gdb -q -batch -ex run -ex "thread apply all bt" --args "${pytest_cmd[@]}")
    ;;
  valgrind)
    if ! command -v valgrind >/dev/null 2>&1; then
      echo "ERROR: valgrind not found." >&2
      exit 127
    fi
    export PYTHONMALLOC=malloc
    runner_cmd=(valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --error-exitcode=1 "${pytest_cmd[@]}")
    ;;
  core)
    ulimit -c unlimited || true
    runner_cmd=("${pytest_cmd[@]}")
    ;;
  *)
    echo "Usage: $0 [pytest|gdb|valgrind|core] [pytest-args...]" >&2
    exit 2
    ;;
esac

if [ "$use_pytest_xvfb" -eq 0 ]; then
  if [ "$has_xvfb_run" -eq 1 ]; then
    runner_cmd=(xvfb-run -a -s "-screen 0 1920x1080x24" "${runner_cmd[@]}")
  else
    if [ "$xvfb_usable" -eq 0 ]; then
      echo "WARN: Xvfb disabled; running without Xvfb." >&2
    else
      echo "WARN: xvfb-run not found; running without Xvfb." >&2
    fi
  fi
fi

if [ "$use_pytest_xvfb" -eq 1 ]; then
  xvfb_note=" + pytest-xvfb"
elif [ "$has_xvfb_run" -eq 1 ]; then
  xvfb_note=" + xvfb-run"
else
  xvfb_note=""
fi
echo "Running GUI Tests (${mode}) using pytest-qt${xvfb_note}..."
if [ "$mode" = "core" ]; then
  echo "Core dumps enabled via ulimit -c unlimited."
fi

"${runner_cmd[@]}"
