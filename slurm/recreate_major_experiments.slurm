#!/bin/bash
#SBATCH --job-name=loc_recreate
#SBATCH --partition=mtech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --output=loc_recreate_%j.out
#SBATCH --error=loc_recreate_%j.err

set -euo pipefail

REPO_ROOT="${LOCALISATION_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
RUN_STAMP="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${REPO_ROOT}/repro_runs/${RUN_STAMP}"
PYTHON_BIN="${LOCALISATION_PYTHON:-python3}"
DEVICE="${LOCALISATION_DEVICE:-auto}"
EPOCH_SCALE="${LOCALISATION_EPOCH_SCALE:-1.0}"
PYTHON_MODULE="${LOCALISATION_PYTHON_MODULE:-python/3.10.pytorch}"
PYTHON_PREFIX="${LOCALISATION_PYTHON_PREFIX:-/opt/ohpc/apps/python/3.10.pytorch}"
INCLUDE_4D="${LOCALISATION_INCLUDE_4D:-0}"
VENV_PATH="${LOCALISATION_VENV:-}"
NOTIFY_URL="https://www.nextgenaischool.in/api/notifyme/send"
NOTIFYME_API_KEY=""

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/artifacts"

exec > "${RUN_ROOT}/logs/pipeline.log" 2>&1

notify_status() {
  local status="$1"
  local summary="$2"
  curl -sS -X POST "${NOTIFY_URL}" \
    -H "Authorization: Bearer ${NOTIFY_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "$(printf '{"title":"Localisation - Slurm Recreation %s","body":"Project: Localisation\nTask: Major experiment recreation on DPU\nStatus: %s\nSummary: %s","actionType":"open_history"}' "${status}" "${status}" "${summary}")" \
    >/dev/null || true
}

on_fail() {
  local exit_code=$?
  notify_status "Failed" "Job ${SLURM_JOB_ID:-manual} failed with exit code ${exit_code}. Check ${RUN_ROOT}/logs/pipeline.log."
  exit "${exit_code}"
}

trap on_fail ERR

echo "repo_root=${REPO_ROOT}"
echo "run_root=${RUN_ROOT}"
echo "node=$(hostname)"
echo "python_bin=${PYTHON_BIN}"
echo "python_module=${PYTHON_MODULE}"
echo "python_prefix=${PYTHON_PREFIX}"
echo "venv_path=${VENV_PATH:-none}"
echo "device=${DEVICE}"
echo "epoch_scale=${EPOCH_SCALE}"
echo "include_4d=${INCLUDE_4D}"
echo "start=$(date --iso-8601=seconds 2>/dev/null || date)"

if [[ -d "${PYTHON_PREFIX}" ]]; then
  export PATH="${PYTHON_PREFIX}/include:${PYTHON_PREFIX}/bin:${PATH}"
  export MANPATH="${PYTHON_PREFIX}/man:${MANPATH:-}"
  export LD_LIBRARY_PATH="${PYTHON_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
echo "python_prefix_export=ok"

if [[ -n "${LOCALISATION_CONDA_ENV:-}" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  conda activate "${LOCALISATION_CONDA_ENV}"
fi

if [[ -n "${VENV_PATH}" ]]; then
  source "${VENV_PATH}/bin/activate"
fi
echo "environment_activate=ok"
echo "python_path=$(command -v "${PYTHON_BIN}")"
"${PYTHON_BIN}" -V

export MPLBACKEND=Agg

deps=(torch matplotlib numpy scipy neurodiffeq seaborn cycler)
if [[ "${INCLUDE_4D}" == "1" ]]; then
  deps+=(neurodiffeq_conditions)
fi

"${PYTHON_BIN}" - "${deps[@]}" <<'PY'
import importlib.util
import sys

mods = sys.argv[1:]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {', '.join(missing)}")
print("Python dependency check passed")
PY
echo "dependency_check=ok"

extra_args=()
if [[ "${INCLUDE_4D}" == "1" ]]; then
  extra_args+=(--include-4d)
fi

"${PYTHON_BIN}" "${REPO_ROOT}/experiments/recreate_major_experiments.py" \
  --group all \
  --device "${DEVICE}" \
  --epoch-scale "${EPOCH_SCALE}" \
  --output-root "${RUN_ROOT}" \
  "${extra_args[@]}" \
  --zip-at-end
echo "runner_exit=ok"

echo "done=$(date --iso-8601=seconds 2>/dev/null || date)"
echo "archive=${RUN_ROOT}.zip"
notify_status "Complete" "Job ${SLURM_JOB_ID:-manual} finished successfully. Logs: ${RUN_ROOT}/logs, archive: ${RUN_ROOT}.zip."
