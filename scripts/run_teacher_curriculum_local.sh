#!/usr/bin/env bash
# Run one stage of the reconstructed Franka teacher curriculum on this machine.
#
# Usage:
#   scripts/run_teacher_curriculum_local.sh v2 [Hydra overrides ...]
#   scripts/run_teacher_curriculum_local.sh v2r <v2-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_teacher_curriculum_local.sh v3c <v2r-checkpoint.pt> [Hydra overrides ...]
#
# Outputs and caches stay under the repository.  Camera rendering is disabled during training because it is not
# part of the state teacher and is unsupported by this machine's current driver; evaluation remains enabled.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {v2|v2r|v3c} [init-checkpoint.pt] [Hydra overrides ...]" >&2
  exit 2
fi

STAGE="$1"
shift
INIT_ARGS=()
case "$STAGE" in
  v2)
    CONFIG="config_repro_v2"
    DEFAULT_NAME="teacher_repro_v2_4096_no_rtx"
    ;;
  v2r)
    CONFIG="config_repro_v2r_long"
    DEFAULT_NAME="teacher_repro_v2r_long_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "v2r requires an existing repro_v2 checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  v3c)
    CONFIG="config_repro_v3c"
    DEFAULT_NAME="teacher_repro_v3c_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "v3c requires an existing repro_v2r_long checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  *)
    echo "unknown stage '$STAGE'; expected v2, v2r, or v3c" >&2
    exit 2
    ;;
esac

export FOOD_ROBOT_ARTIFACTS="$ROOT/artifacts"
export XDG_CACHE_HOME="$ROOT/.cache"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

exec "$ROOT/.venv/bin/python" -X faulthandler pipeline/0_state_teacher/train.py \
  --config-name "$CONFIG" \
  env.num_envs=4096 \
  run.name="$DEFAULT_NAME" \
  worker.render=false \
  logger.backend=null \
  "${INIT_ARGS[@]}" \
  "$@"
