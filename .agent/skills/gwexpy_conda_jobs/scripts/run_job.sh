#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_job.sh start <ruff|mypy|pytest> [args...]
  run_job.sh list
  run_job.sh tail <session-name|latest-ruff|latest-mypy|latest-pytest>
  run_job.sh attach <session-name|latest-ruff|latest-mypy|latest-pytest>

Examples:
  run_job.sh start ruff
  run_job.sh start mypy
  run_job.sh start pytest tests/test_import_order.py -q
  run_job.sh list
  run_job.sh tail latest-pytest
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SKILL_DIR}/../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/.agent/tmp/gwexpy_conda_jobs"

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

resolve_session_ref() {
    local ref="$1"
    if [[ -f "${LOG_DIR}/${ref}" ]]; then
        cat "${LOG_DIR}/${ref}"
        return
    fi
    printf '%s\n' "$ref"
}

build_command() {
    local tool="$1"
    shift || true

    case "$tool" in
        ruff)
            printf 'conda run -n gwexpy ruff check'
            if (($# == 0)); then
                printf ' .'
            fi
            ;;
        mypy)
            printf 'conda run -n gwexpy mypy'
            if (($# == 0)); then
                printf ' gwexpy'
            fi
            ;;
        pytest)
            printf 'conda run -n gwexpy pytest'
            ;;
        *)
            echo "Unsupported tool: $tool" >&2
            exit 1
            ;;
    esac

    if (($# > 0)); then
        printf ' '
        printf '%q ' "$@"
    fi
}

start_job() {
    local tool="$1"
    shift || true

    require_cmd tmux
    require_cmd conda
    mkdir -p "$LOG_DIR"

    local ts session log_file cmd
    ts="$(date +%Y%m%d-%H%M%S)"
    session="gwexpy-${tool}-${ts}"
    log_file="${LOG_DIR}/${session}.log"
    cmd="$(build_command "$tool" "$@")"

    tmux new-session -d -s "$session" "cd ${REPO_ROOT} && ${cmd} 2>&1 | tee ${log_file}"
    printf '%s\n' "$session" > "${LOG_DIR}/latest-${tool}"

    cat <<EOF
Started:
  session: ${session}
  log: ${log_file}
  command: ${cmd}

Next:
  bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh tail ${session}
  bash .agent/skills/gwexpy_conda_jobs/scripts/run_job.sh attach ${session}
EOF
}

list_jobs() {
    require_cmd tmux
    mkdir -p "$LOG_DIR"
    tmux ls 2>/dev/null | grep '^gwexpy-' || true
}

tail_job() {
    local session
    session="$(resolve_session_ref "$1")"
    local log_file="${LOG_DIR}/${session}.log"
    if [[ ! -f "$log_file" ]]; then
        echo "Log file not found: $log_file" >&2
        exit 1
    fi
    tail -n 50 "$log_file"
}

attach_job() {
    require_cmd tmux
    local session
    session="$(resolve_session_ref "$1")"
    exec tmux attach -t "$session"
}

main() {
    if (($# == 0)); then
        usage
        exit 1
    fi

    local subcommand="$1"
    shift

    case "$subcommand" in
        start)
            if (($# < 1)); then
                usage
                exit 1
            fi
            start_job "$@"
            ;;
        list)
            list_jobs
            ;;
        tail)
            if (($# != 1)); then
                usage
                exit 1
            fi
            tail_job "$1"
            ;;
        attach)
            if (($# != 1)); then
                usage
                exit 1
            fi
            attach_job "$1"
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

main "$@"
