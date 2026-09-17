#!/usr/bin/env bash
set -euo pipefail

export PATH="${PATH:-/usr/bin:/bin}:/usr/local/bin:/opt/homebrew/bin:/usr/sbin:/sbin"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/repro_runs/gaussian_validation/project}"
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
    "Run root: {root}; ok={ok}; failures={fail}; seeds={seeds}".format(
        root=batch_root,
        ok=counts.get("ok", 0),
        fail=counts.get("failures", 0),
        seeds=counts.get("seeds", 0),
    )
)
PY
}

zip_batch() {
  if ! command -v zip >/dev/null 2>&1; then
    return 0
  fi
  local parent_dir archive_name batch_name
  parent_dir="$(dirname "${BATCH_ROOT}")"
  batch_name="$(basename "${BATCH_ROOT}")"
  archive_name="${batch_name}.zip"
  (
    cd "${parent_dir}"
    rm -f "${archive_name}"
    zip -qr "${archive_name}" "${batch_name}"
  )
}

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export LOCALISATION_TORCH_THREADS="${LOCALISATION_TORCH_THREADS:-1}"

LOGICAL_CPU="$(sysctl -n hw.logicalcpu 2>/dev/null || getconf _NPROCESSORS_ONLN || echo 10)"
DEFAULT_MAX_WORKERS=8
if [[ "${LOGICAL_CPU}" =~ ^[0-9]+$ ]]; then
  if (( LOGICAL_CPU <= 2 )); then
    DEFAULT_MAX_WORKERS=1
  elif (( LOGICAL_CPU - 2 < 8 )); then
    DEFAULT_MAX_WORKERS="$((LOGICAL_CPU - 2))"
  fi
fi

MAX_WORKERS="${LOCALISATION_MAX_WORKERS:-${DEFAULT_MAX_WORKERS}}"
DEVICE="${LOCALISATION_DEVICE:-cpu}"

if [[ -n "${LOCALISATION_CV_SEEDS:-}" ]]; then
  read -r -a SEEDS <<<"${LOCALISATION_CV_SEEDS}"
else
  SEEDS=(1 2 3 4 5 6 7 8 9 10)
fi

ARGS=(
  --max-workers "${MAX_WORKERS}"
  --device "${DEVICE}"
  --output-root "${BATCH_ROOT}"
  --seeds "${SEEDS[@]}"
)

if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

echo "Batch root: ${BATCH_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Args: ${ARGS[*]}"

rc=0
if "${PYTHON_BIN}" "${ROOT_DIR}/experiments/validate_localisation_matrix.py" "${ARGS[@]}"; then
  zip_batch || true
else
  rc=$?
  zip_batch || true
  echo "Gaussian validation failed: $(summary_line) (exit=${rc})" >&2
  exit "${rc}"
fi
