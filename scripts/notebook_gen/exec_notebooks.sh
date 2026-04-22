#!/usr/bin/env bash
set -euo pipefail

NOTEBOOK_GLOB="${1:-docs/web/ja/user_guide/tutorials/*.ipynb}"
LOG_ROOT="${2:-temp_logs/notebook_exec}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/exec_${TIMESTAMP}.log"

mkdir -p "${LOG_ROOT}"

{
  printf 'Starting notebook execution at %s\n' "$(date --iso-8601=seconds)"
  printf 'Notebook glob: %s\n' "${NOTEBOOK_GLOB}"
  printf 'Log file: %s\n' "${LOG_FILE}"
} > "${LOG_FILE}"

shopt -s nullglob
mapfile -t FILES < <(compgen -G "${NOTEBOOK_GLOB}" || true)
shopt -u nullglob

if [ "${#FILES[@]}" -eq 0 ]; then
  printf 'No notebooks matched: %s\n' "${NOTEBOOK_GLOB}" | tee -a "${LOG_FILE}"
  exit 1
fi

for nb in "${FILES[@]}"; do
  printf '\n[%s] Executing %s\n' "$(date --iso-8601=seconds)" "${nb}" | tee -a "${LOG_FILE}"
  conda run -n gwexpy jupyter nbconvert --to notebook --execute --inplace "${nb}" >> "${LOG_FILE}" 2>&1
done

printf '\nCompleted notebook execution at %s\n' "$(date --iso-8601=seconds)" | tee -a "${LOG_FILE}"
