#!/usr/bin/env bash
set -euo pipefail

export PATH="${PATH:-/usr/bin:/bin}:/usr/local/bin:/opt/homebrew/bin:/usr/sbin:/sbin"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/repro_runs/project}"
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

BATCH_ID="${BATCH_ID:-$(next_batch_id)}"
BATCH_ROOT="${BATCH_ROOT:-${BASE_RUN_ROOT}/${BATCH_ID}}"
mkdir -p "${BATCH_ROOT}"

NOTIFYME_ENDPOINT="${NOTIFYME_ENDPOINT:-https://www.nextgenaischool.in/api/notifyme/send}"
NOTIFYME_API_KEY="${NOTIFYME_API_KEY:-NOTIFYME_API_KEY}"
SCRIPT_COMPLETED=0
STOP_NOTIFIED=0

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

  echo "No Python interpreter found for LPINNs." >&2
  return 1
}

PYTHON_BIN="$(resolve_python_bin)"

send_notify() {
  local title="$1"
  local body="$2"

  if [[ -z "${NOTIFYME_API_KEY}" ]] || ! command -v curl >/dev/null 2>&1; then
    return 0
  fi

  curl -fsS -m 20 -X POST "${NOTIFYME_ENDPOINT}" \
    -H "Authorization: Bearer ${NOTIFYME_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"title\":\"${title}\",\"body\":\"${body}\",\"actionType\":\"open_history\"}" >/dev/null || true
}

summary_line() {
  local summary_path="${BATCH_ROOT}/summary.json"
  "${PYTHON_BIN}" - "${summary_path}" "${BATCH_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
batch_root = Path(sys.argv[2])
if not summary_path.exists():
    print(f"Summary unavailable. Run root: {batch_root}")
    raise SystemExit(0)

payload = json.loads(summary_path.read_text())
counts = payload.get("counts", {})
print(
    "Run root: {root}; ok={ok}; failures={fail}; allowed_failures={allowed}".format(
        root=batch_root,
        ok=counts.get("ok", 0),
        fail=counts.get("failures", 0),
        allowed=counts.get("allowed_failures", 0),
    )
)
PY
}

notify_stopped_once() {
  if [[ "${STOP_NOTIFIED}" -eq 1 ]]; then
    return 0
  fi
  STOP_NOTIFIED=1
  send_notify "LPINNs - Task Failed" "Project: LPINNs
Task: Full local experiment recreation
Status: Failed
Error: Run stopped before completion. Run root: ${BATCH_ROOT}"
}

on_interrupt() {
  notify_stopped_once
  exit 130
}

on_exit() {
  local rc=$?
  if [[ "${rc}" -ne 0 && "${SCRIPT_COMPLETED}" -eq 0 ]]; then
    notify_stopped_once
  fi
}

trap on_interrupt INT TERM
trap on_exit EXIT

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export LOCALISATION_TORCH_THREADS="${LOCALISATION_TORCH_THREADS:-1}"

MAX_WORKERS="${LOCALISATION_MAX_WORKERS:-8}"
DEVICE="${LOCALISATION_DEVICE:-cpu}"

ARGS=(
  --group all
  --include-4d
  --include-blocked
  --max-workers "${MAX_WORKERS}"
  --device "${DEVICE}"
  --output-root "${BATCH_ROOT}"
  --zip-at-end
)

if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

echo "Batch root: ${BATCH_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Args: ${ARGS[*]}"

rc=0
if "${PYTHON_BIN}" "${ROOT_DIR}/experiments/recreate_major_experiments.py" "${ARGS[@]}"; then
  SCRIPT_COMPLETED=1
  send_notify "LPINNs - Task Completion" "Project: LPINNs
Task: Full local experiment recreation
Status: Success
Summary: $(summary_line)"
else
  rc=$?
  send_notify "LPINNs - Task Failed" "Project: LPINNs
Task: Full local experiment recreation
Status: Failed
Error: $(summary_line) (exit=${rc})"
  exit "${rc}"
fi
