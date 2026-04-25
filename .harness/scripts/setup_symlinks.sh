#!/usr/bin/env bash
# setup_symlinks.sh — Set up AI tool symlinks pointing to .harness/ for a project.
#
# Usage:
#   bash setup_symlinks.sh [--dry-run] [--tools claude,codex,cursor,gemini]
#
# This script is copied to <project>/.harness/scripts/setup_symlinks.sh
# and executed from the project root.
#
# .harness.local/ support:
#   If .harness.local/ exists alongside .harness/, its subdirectories take
#   precedence over the corresponding .harness/ subdirectories for Claude.
#   This allows per-developer overrides (e.g. different conda env names in
#   hooks.json) without committing local settings to the repository.
#   .harness.local/ is listed in .gitignore and must never be committed.
set -euo pipefail

# Resolve project root (two levels up from this script's location: .harness/scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HARNESS_DIR="$PROJECT_ROOT/.harness"
HARNESS_LOCAL_DIR="$PROJECT_ROOT/.harness.local"
DRY_RUN=false
TOOLS="claude,codex,cursor,gemini"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --tools) TOOLS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--tools claude,codex,cursor,gemini]"
            exit 0
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_ok()   { echo -e "${GREEN}[OK]${NC}   $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; }
log_dry()  { echo -e "${YELLOW}[DRY]${NC}  $1"; }
log_err()  { echo -e "${RED}[ERR]${NC}  $1"; }

# make_symlink TARGET LINK
#   TARGET: relative path from LINK's parent directory (e.g. "../../.harness/hooks")
#   LINK:   absolute path to the symlink to create
make_symlink() {
    local target="$1"
    local link="$2"
    local link_dir
    link_dir="$(dirname "$link")"

    if [[ "$DRY_RUN" == true ]]; then
        log_dry "ln -sfn $target $link"
        return 0
    fi

    mkdir -p "$link_dir"

    # Already points to the correct target — skip
    if [[ -L "$link" ]] && [[ "$(readlink "$link")" == "$target" ]]; then
        log_skip "Already linked: $link -> $target"
        return 0
    fi

    # Back up an existing real directory so we don't lose data
    if [[ -d "$link" ]] && [[ ! -L "$link" ]]; then
        local backup="${link}.backup-$(date +%Y%m%d%H%M%S)"
        mv "$link" "$backup"
        log_warn "Backed up existing directory: $link -> $backup"
    fi

    ln -sfn "$target" "$link"
    log_ok "Linked: $link -> $target"
}

setup_claude() {
    log_ok "Setting up Claude symlinks..."
    local base="$PROJECT_ROOT/.claude"

    # .harness.local/ takes precedence when a subdirectory exists there.
    _claude_link() {
        local subdir="$1"
        if [[ -d "$HARNESS_LOCAL_DIR/$subdir" ]]; then
            make_symlink "../../.harness.local/$subdir" "$base/$subdir"
            log_warn "  Using local override: .harness.local/$subdir"
        elif [[ -d "$HARNESS_DIR/$subdir" ]]; then
            make_symlink "../../.harness/$subdir" "$base/$subdir"
        fi
    }

    _claude_link hooks
    _claude_link agents
    _claude_link workflows
    _claude_link rules
}

setup_codex() {
    local base="$PROJECT_ROOT/.codex"
    if ! command -v codex &>/dev/null && [[ ! -d "$base" ]]; then
        log_skip "Codex: not detected, skipping"
        return 0
    fi
    log_ok "Setting up Codex symlinks..."
    [[ -d "$HARNESS_DIR/hooks" ]]  && make_symlink "../../.harness/hooks"  "$base/hooks"
    [[ -d "$HARNESS_DIR/agents" ]] && make_symlink "../../.harness/agents" "$base/agents"
}

setup_cursor() {
    local base="$PROJECT_ROOT/.cursor"
    if [[ ! -d "$base" ]] && ! command -v cursor &>/dev/null; then
        log_skip "Cursor: not detected, skipping"
        return 0
    fi
    log_ok "Setting up Cursor symlinks..."
    [[ -d "$HARNESS_DIR/agents" ]] && make_symlink "../../.harness/agents" "$base/agents"
}

setup_gemini() {
    local base="$PROJECT_ROOT/.gemini"
    if [[ ! -d "$base" ]] && ! command -v gemini &>/dev/null; then
        log_skip "Gemini: not detected, skipping"
        return 0
    fi
    log_ok "Setting up Gemini symlinks..."
    [[ -d "$HARNESS_DIR/agents" ]] && make_symlink "../../.harness/agents" "$base/agents"
}

verify_all() {
    echo ""
    echo "Verifying symlinks..."
    local all_ok=true
    local targets=(
        "$PROJECT_ROOT/.claude/hooks:../.harness/hooks"
        "$PROJECT_ROOT/.claude/agents:../.harness/agents"
        "$PROJECT_ROOT/.claude/workflows:../.harness/workflows"
    )
    for entry in "${targets[@]}"; do
        local link="${entry%%:*}"
        local expected_target="${entry##*:}"
        if [[ -L "$link" ]]; then
            log_ok "OK: $link -> $(readlink "$link")"
        elif [[ ! -d "${link%/*}/.harness/${link##*/.harness/}" ]]; then
            log_skip "No source dir for $link (skipped during setup)"
        else
            log_warn "Missing symlink: $link"
            all_ok=false
        fi
    done
    echo ""
    if [[ "$all_ok" == true ]]; then
        log_ok "All symlinks verified."
    else
        log_warn "Some symlinks are missing. Re-run without --dry-run."
    fi
}

main() {
    echo "Project Harness Symlink Setup"
    echo "Project root : $PROJECT_ROOT"
    echo "Tools        : $TOOLS"
    [[ "$DRY_RUN" == true ]] && echo "Mode         : DRY-RUN (no changes will be made)"
    echo ""

    IFS=',' read -ra TOOL_LIST <<< "$TOOLS"
    for tool in "${TOOL_LIST[@]}"; do
        case "$tool" in
            claude)  setup_claude  ;;
            codex)   setup_codex   ;;
            cursor)  setup_cursor  ;;
            gemini)  setup_gemini  ;;
            *) log_warn "Unknown tool: $tool (skipping)" ;;
        esac
    done

    if [[ "$DRY_RUN" == false ]]; then
        verify_all
    fi
}

main "$@"
