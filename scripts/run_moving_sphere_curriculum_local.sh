#!/usr/bin/env bash
# Run the independent moving-sphere/static-bowl curriculum on the local 3090.
# Usage:
#   scripts/run_moving_sphere_curriculum_local.sh g0s0c [Hydra overrides ...]
#   scripts/run_moving_sphere_curriculum_local.sh g0s0d <g0s0-best.pt> [Hydra overrides ...]
#   scripts/run_moving_sphere_curriculum_local.sh g0s0e [Hydra overrides ...]
#   scripts/run_moving_sphere_curriculum_local.sh g0s1 <g0s0e-best.pt> [Hydra overrides ...]
#   scripts/run_moving_sphere_curriculum_local.sh g0s1b <g0s1-best.pt> [Hydra overrides ...]
#   scripts/run_moving_sphere_curriculum_local.sh g0s1c <g0s1-best.pt> [Hydra overrides ...]
#   scripts/run_moving_sphere_curriculum_local.sh g0s2 <g0s1-best.pt> [Hydra overrides ...]
#   scripts/run_moving_sphere_curriculum_local.sh g0s2c <g0s1c-final.pt> [Hydra overrides ...]
#   scripts/run_moving_sphere_curriculum_local.sh g0s2d <g0s2c-best.pt> [Hydra overrides ...]
#   scripts/run_moving_sphere_curriculum_local.sh g1 <g0s2c-best.pt> [Hydra overrides ...]
#   scripts/run_moving_sphere_curriculum_local.sh g0s0 [Hydra overrides ...]  # historical multi-yaw run
#   scripts/run_moving_sphere_curriculum_local.sh s0 [Hydra overrides ...]  # historical
#   scripts/run_moving_sphere_curriculum_local.sh s1 <s0-best.pt> [Hydra overrides ...]
#   scripts/run_moving_sphere_curriculum_local.sh s2 <s1-best.pt> [Hydra overrides ...]
#   scripts/run_moving_sphere_curriculum_local.sh s3 <s2-best.pt> [Hydra overrides ...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {g0s0c|g0s0d|g0s0e|g0s1|g0s1b|g0s1c|g0s2|g0s2c|g0s2d|g1|g0s0|s0|s1|s2|s3} [init-checkpoint.pt] [Hydra overrides ...]" >&2
  exit 2
fi

STAGE="$1"
shift
INIT_ARGS=()
case "$STAGE" in
  g0s0c)
    CONFIG="config_moving_sphere_g0_s0_canonical_gate"
    DEFAULT_NAME="moving_sphere_g0_s0c_canonical_gate_1024_no_rtx"
    NUM_ENVS=1024
    ;;
  g0s0d)
    CONFIG="config_moving_sphere_g0_s0d_canonical_coarse"
    DEFAULT_NAME="moving_sphere_g0_s0d_canonical_coarse_1024_no_rtx"
    NUM_ENVS=1024
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "g0s0d requires the selected historical G0-S0 coarse checkpoint" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  g0s0e)
    CONFIG="config_moving_sphere_g0_s0e_residual_baseline"
    DEFAULT_NAME="moving_sphere_g0_s0e_residual_baseline_1024_no_rtx"
    NUM_ENVS=1024
    ;;
  g0s1)
    CONFIG="config_moving_sphere_g0_s1_speed_001_003"
    DEFAULT_NAME="moving_sphere_g0_s1_speed_001_003_1024_no_rtx"
    NUM_ENVS=1024
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "g0s1 requires the selected G0-S0e checkpoint" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  g0s1b)
    CONFIG="config_moving_sphere_g0_s1b_speed_002_004"
    DEFAULT_NAME="moving_sphere_g0_s1b_speed_002_004_1024_no_rtx"
    NUM_ENVS=1024
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "g0s1b requires the selected G0-S1 checkpoint" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  g0s1c)
    CONFIG="config_moving_sphere_g0_s1c_early_intercept_002_004"
    DEFAULT_NAME="moving_sphere_g0_s1c_early_intercept_002_004_1024_no_rtx"
    NUM_ENVS=1024
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "g0s1c requires the selected G0-S1 checkpoint for a fair interceptor A/B comparison" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  g0s2)
    CONFIG="config_moving_sphere_g0_s2_speed_003_005"
    DEFAULT_NAME="moving_sphere_g0_s2_speed_003_005_1024_no_rtx"
    NUM_ENVS=1024
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "g0s2 requires the selected G0-S1 checkpoint" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  g0s2c)
    CONFIG="config_moving_sphere_g0_s2c_align_follow_speed_008"
    DEFAULT_NAME="moving_sphere_g0_s2c_align_follow_speed_008_1024_no_rtx"
    NUM_ENVS=1024
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "g0s2c requires the frozen G0-S1c final checkpoint" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  g0s2d)
    CONFIG="config_moving_sphere_g0_s2d_position_randomization_008"
    DEFAULT_NAME="moving_sphere_g0_s2d_position_randomization_008_1024_no_rtx"
    NUM_ENVS=1024
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "g0s2d requires the selected G0-S2c checkpoint" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  g1)
    CONFIG="config_moving_sphere_g1_place_008"
    DEFAULT_NAME="moving_sphere_g1_place_008_1024_no_rtx"
    NUM_ENVS=1024
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "g1 requires the frozen selected G0-S2c checkpoint" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")" "init_actor_scale=-4.0")
    shift
    ;;
  g0s0)
    CONFIG="config_moving_sphere_g0_s0_ppo"
    DEFAULT_NAME="moving_sphere_g0_s0_ppo_1024_no_rtx"
    NUM_ENVS=1024
    ;;
  s0)
    CONFIG="config_moving_sphere_s0_lift"
    DEFAULT_NAME="moving_sphere_s0_lift_4096_no_rtx"
    NUM_ENVS=4096
    ;;
  s1|s2|s3)
    case "$STAGE" in
      s1) CONFIG="config_moving_sphere_s1_place"; DEFAULT_NAME="moving_sphere_s1_place_4096_no_rtx"; NUM_ENVS=4096 ;;
      s2) CONFIG="config_moving_sphere_s2_speed"; DEFAULT_NAME="moving_sphere_s2_speed_4096_no_rtx"; NUM_ENVS=4096 ;;
      s3) CONFIG="config_moving_sphere_s3_home"; DEFAULT_NAME="moving_sphere_s3_home_4096_no_rtx"; NUM_ENVS=4096 ;;
    esac
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "$STAGE requires the selected checkpoint from the preceding sphere stage" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  *)
    echo "unknown stage '$STAGE'; expected g0s0c, g0s0d, g0s0e, g0s1, g0s1b, g0s1c, g0s2, g0s2c, g0s2d, g1, g0s0, s0, s1, s2, or s3" >&2
    exit 2
    ;;
esac

export FOOD_ROBOT_ARTIFACTS="$REPO_ROOT/artifacts/moving_sphere_static_bowl"
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
# Host robot's physical GPU0 is retired because of its cooling fault.  Isaac
# sees physical GPU1 as logical cuda:0 inside this process.
if [[ "$(hostname)" == "robot" ]]; then
  export PICKPLACE_PHYSICAL_GPU=1
fi

exec "$REPO_ROOT/.venv/bin/python" -X faulthandler pipeline/0_state_teacher/train.py \
  --config-name "$CONFIG" \
  env.num_envs="${SPHERE_NUM_ENVS:-$NUM_ENVS}" \
  run.name="$DEFAULT_NAME" \
  worker.render=false \
  logger.backend=null \
  "${INIT_ARGS[@]}" \
  "$@"
