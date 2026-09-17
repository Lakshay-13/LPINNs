#!/usr/bin/env bash
set -euo pipefail

export PATH="${PATH:-/usr/bin:/bin}:/usr/local/bin:/opt/homebrew/bin:/usr/sbin:/sbin"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/repro_runs/project}"

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

usage() {
  cat <<'EOF'
Usage:
  run_all_experiments_detached.sh start [run_all_experiments args...]
  run_all_experiments_detached.sh status <screen_name>
  run_all_experiments_detached.sh stop <screen_name>
EOF
}

screen_exists() {
  local screen_name="${1:?screen_name required}"
  local listing
  listing="$(screen -ls 2>/dev/null || true)"
  grep -E "[[:space:]][0-9]+\\.${screen_name}[[:space:]]" <<<"${listing}" >/dev/null 2>&1
}

start_job() {
  if ! command -v screen >/dev/null 2>&1; then
    echo "screen is required but not installed." >&2
    return 1
  fi

  local batch_id="${BATCH_ID:-$(next_batch_id)}"
  local screen_name="${SCREEN_NAME:-localisation_${batch_id}}"
  local batch_root="${BATCH_ROOT:-${BASE_RUN_ROOT}/${batch_id}}"
  local launcher_log="${LAUNCH_LOG:-${batch_root}/launcher.log}"

  if screen_exists "${screen_name}"; then
    echo "screen session already exists: ${screen_name}" >&2
    return 1
  fi

  mkdir -p "${batch_root}"

  local -a run_args=("$@")
  local quoted_root quoted_batch_id quoted_batch_root quoted_launcher_log quoted_args cmd
  printf -v quoted_root "%q" "${ROOT_DIR}"
  printf -v quoted_batch_id "%q" "${batch_id}"
  printf -v quoted_batch_root "%q" "${batch_root}"
  printf -v quoted_launcher_log "%q" "${launcher_log}"
  quoted_args=""
  if (( ${#run_args[@]} > 0 )); then
    printf -v quoted_args "%q " "${run_args[@]}"
  fi

  cmd="cd ${quoted_root} && export BATCH_ID=${quoted_batch_id} BATCH_ROOT=${quoted_batch_root} && exec /bin/bash ./run_all_experiments.sh ${quoted_args} >> ${quoted_launcher_log} 2>&1"
  screen -dmS "${screen_name}" /bin/bash -lc "${cmd}"

  echo "SCREEN_NAME=${screen_name}"
  echo "BATCH_ID=${batch_id}"
  echo "BATCH_ROOT=${batch_root}"
  echo "LAUNCH_LOG=${launcher_log}"
}

status_job() {
  local screen_name="${1:?screen_name required}"
  local listing
  listing="$(screen -ls 2>/dev/null || true)"
  grep "${screen_name}" <<<"${listing}" || {
    echo "No screen session found for ${screen_name}" >&2
    return 1
  }
}

stop_job() {
  local screen_name="${1:?screen_name required}"
  screen -S "${screen_name}" -X quit
}

main() {
  local action="${1:-start}"
  case "${action}" in
    start)
      shift || true
      start_job "$@"
      ;;
    status)
      shift
      status_job "$@"
      ;;
    stop)
      shift
      stop_job "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      usage >&2
      return 1
      ;;
  esac
}

main "$@"
