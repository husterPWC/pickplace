#!/usr/bin/env bash
# Independent ZU5+AG95 moving-sphere curriculum on the local 3090.
# Physical GPU0 is never used on host `robot`; Isaac sees physical GPU1 as
# logical cuda:0 through PICKPLACE_PHYSICAL_GPU=1.
#
# Usage:
#   scripts/run_jaka_sphere_curriculum_local.sh j0s0 [Hydra overrides ...]
#   scripts/run_jaka_sphere_curriculum_local.sh j0s1 <j0s0-best.pt> [Hydra overrides ...]
#   scripts/run_jaka_sphere_curriculum_local.sh j0s2 <j0s1-best.pt> [Hydra overrides ...]
#   scripts/run_jaka_sphere_curriculum_local.sh j0s3a <accepted-j0s2-actor.pt> [Hydra overrides ...]
#   scripts/run_jaka_sphere_curriculum_local.sh j0s3b <accepted-j0s3a-actor.pt> [Hydra overrides ...]
#   scripts/run_jaka_sphere_curriculum_local.sh j1 <accepted-j0s3b-actor.pt> [Hydra overrides ...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {j0s0|j0s1|j0s2|j0s3a|j0s3b|j1} [init-checkpoint.pt] [Hydra overrides ...]" >&2
  exit 2
fi

STAGE="$1"
shift
INIT_ARGS=()
case "$STAGE" in
  j0s0)
    CONFIG="config_jaka_sphere_j0_s0_residual_001_003"
    DEFAULT_NAME="jaka_zu5_ag95_sphere_j0_s0_residual_001_003_1024_no_rtx"
    NUM_ENVS=1024
    ;;
  j0s1)
    CONFIG="config_jaka_sphere_j0_s1_residual_003_005"
    DEFAULT_NAME="jaka_zu5_ag95_sphere_j0_s1_residual_003_005_1024_no_rtx"
    NUM_ENVS=1024
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "j0s1 requires the accepted JAKA J0-S0 checkpoint" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  j0s2)
    CONFIG="config_jaka_sphere_j0_s2_speed_008"
    DEFAULT_NAME="jaka_zu5_ag95_sphere_j0_s2_speed008_1024_no_rtx"
    NUM_ENVS=1024
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "j0s2 requires the accepted JAKA J0-S1 checkpoint" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  j0s3a)
    CONFIG="config_jaka_sphere_j0_s3a_position_randomization_008"
    DEFAULT_NAME="jaka_zu5_ag95_sphere_j0_s3a_position_randomization_008_1024_no_rtx"
    NUM_ENVS=1024
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "j0s3a requires the accepted JAKA J0-S2 actor checkpoint" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  j0s3b)
    CONFIG="config_jaka_sphere_j0_s3b_speed_006_008"
    DEFAULT_NAME="jaka_zu5_ag95_sphere_j0_s3b_speed_006_008_1024_no_rtx"
    NUM_ENVS=1024
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "j0s3b requires the accepted JAKA J0-S3a actor checkpoint" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  j1)
    CONFIG="config_jaka_sphere_j1_place_006_008"
    DEFAULT_NAME="jaka_zu5_ag95_sphere_j1_place_006_008_1024_no_rtx"
    NUM_ENVS=1024
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "j1 requires the accepted JAKA J0-S3b actor checkpoint" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  *)
    echo "unknown stage '$STAGE'; expected j0s0, j0s1, j0s2, j0s3a, j0s3b, or j1" >&2
    exit 2
    ;;
esac

export FOOD_ROBOT_ARTIFACTS="$REPO_ROOT/artifacts/jaka_zu5_ag95_sphere"
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
if [[ "$(hostname)" == "robot" ]]; then
  export PICKPLACE_PHYSICAL_GPU=1
fi

exec "$REPO_ROOT/.venv/bin/python" -X faulthandler pipeline/0_state_teacher/train.py \
  --config-name "$CONFIG" \
  env.num_envs="${JAKA_NUM_ENVS:-$NUM_ENVS}" \
  run.name="$DEFAULT_NAME" \
  worker.render=false \
  logger.backend=null \
  "${INIT_ARGS[@]}" \
  "$@"
