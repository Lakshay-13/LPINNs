#!/usr/bin/env bash
set -euo pipefail

export PATH="${PATH:-/usr/bin:/bin}:/usr/local/bin:/opt/homebrew/bin:/usr/sbin:/sbin"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/campaign_runs/project}"
mkdir -p "${BASE_RUN_ROOT}"

next_batch_id() {
  local max_run=0
  local entry suffix
  shopt -s nullglob
  for entry in "${BASE_RUN_ROOT}"/run_*; do
    [[ -d "${entry}" ]] || continue
    suffix="${entry##*/run_}"
    [[ "${suffix}" =~ ^[0-9]+$ ]] || continue
    if (( suffix > max_run )); then
      max_run="${suffix}"
    fi
  done
  shopt -u nullglob
  printf 'run_%s\n' "$((max_run + 1))"
}

resolve_python_bin() {
  if [[ -n "${LOCALISATION_PYTHON:-}" ]]; then
    printf '%s\n' "${LOCALISATION_PYTHON}"
    return 0
  fi

  local candidate
  for candidate in "${ROOT_DIR}/.venv/bin/python" python3.11 python3 python; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    if command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done

  echo "No Python interpreter found for the localisation campaign." >&2
  return 1
}

has_flag() {
  local needle="${1:?flag required}"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(resolve_python_bin)"
BATCH_ID="${BATCH_ID:-$(next_batch_id)}"
BATCH_ROOT="${BATCH_ROOT:-${BASE_RUN_ROOT}/${BATCH_ID}}"
mkdir -p "${BATCH_ROOT}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export LOCALISATION_TORCH_THREADS="${LOCALISATION_TORCH_THREADS:-1}"

declare -a ARGS
ARGS=(--output-root "${BATCH_ROOT}")
if ! has_flag "--max-workers" "$@"; then
  ARGS+=(--max-workers "${LOCALISATION_MAX_WORKERS:-10}")
fi
if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

echo "Batch root: ${BATCH_ROOT}"
echo "Batch id: ${BATCH_ID}"
echo "Python: ${PYTHON_BIN}"
echo "Args: ${ARGS[*]}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/experiments/run_localisation_campaign.py" "${ARGS[@]}"
