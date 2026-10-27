#!/usr/bin/env bash
# Run one independent moving-cube curriculum stage on the local 3090.
#
# Usage:
#   scripts/run_moving_cube_curriculum_local.sh c0a <grasp-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c0s <c0a-best-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c0k <c0a-best-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c0h <c0a-best-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c0i <c0a-best-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c0m <c0a-best-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c0n <c0m-best-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c0o <c0m-best-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c0t [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c0u [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c0v <c0u-best-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c0w <c0u-best-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c0x <c0u-best-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c0b <c0a-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c1 [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c2 <c1-checkpoint.pt> [Hydra overrides ...]
#   scripts/run_moving_cube_curriculum_local.sh c3h <c2-checkpoint.pt> [Hydra overrides ...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 {c0a|c0s|c0k|c0h|c0i|c0m|c0n|c0o|c0t|c0u|c0v|c0w|c0x|c0b|c1|c2|c3h} [init-checkpoint.pt] [Hydra overrides ...]" >&2
  exit 2
fi

STAGE="$1"
shift
INIT_ARGS=()
case "$STAGE" in
  c0|c0a)
    CONFIG="config_moving_cube_c0_lift"
    DEFAULT_NAME="moving_cube_c0a_lift3cm_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c0a requires an existing grasp-capable checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  c0s)
    CONFIG="config_moving_cube_c0s_stabilize"
    DEFAULT_NAME="moving_cube_c0s_stabilize3cm_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c0s requires the best C0a checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  c0k)
    CONFIG="config_moving_cube_c0k_reference_kl"
    DEFAULT_NAME="moving_cube_c0k_reference_kl_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c0k requires the best C0a checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  c0h)
    CONFIG="config_moving_cube_c0h_hold_stability"
    DEFAULT_NAME="moving_cube_c0h_hold6_slip10_probe_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c0h requires the best C0a checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  c0i)
    CONFIG="config_moving_cube_c0i_staged_hold"
    DEFAULT_NAME="moving_cube_c0i_hold5_prelift2cm_probe_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c0i requires the best C0a checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  c0m)
    CONFIG="config_moving_cube_c0m_temporal"
    DEFAULT_NAME="moving_cube_c0m_temporal_geometry_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c0m requires the best C0a checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  c0n)
    CONFIG="config_moving_cube_c0n_stable_grasp_target"
    DEFAULT_NAME="moving_cube_c0n_stable_grasp_target_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c0n requires the selected C0m checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  c0o)
    CONFIG="config_moving_cube_c0o_velocity_coupling"
    DEFAULT_NAME="moving_cube_c0o_velocity_coupling_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c0o requires the selected C0m checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  c0t)
    CONFIG="config_moving_cube_c0t_gap_direct_rl"
    DEFAULT_NAME="moving_cube_c0t_gap_direct_rl_4096_no_rtx"
    ;;
  c0u)
    CONFIG="config_moving_cube_c0u_close_direct_rl"
    DEFAULT_NAME="moving_cube_c0u_close_direct_rl_4096_no_rtx"
    ;;
  c0v)
    CONFIG="config_moving_cube_c0v_lift_direct_rl"
    DEFAULT_NAME="moving_cube_c0v_lift_direct_rl_probe20m_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c0v requires the selected C0u stable-close checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  c0w)
    CONFIG="config_moving_cube_c0w_micro_lift_bridge"
    DEFAULT_NAME="moving_cube_c0w_micro_lift_bridge_probe10m_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c0w requires the selected C0u stable-close checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  c0x)
    CONFIG="config_moving_cube_c0x_gated_micro_lift"
    DEFAULT_NAME="moving_cube_c0x_gated_micro_lift_probe6m_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c0x requires the selected C0u stable-close checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  c0b)
    CONFIG="config_moving_cube_c0b_lift"
    DEFAULT_NAME="moving_cube_c0b_lift10cm_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c0b requires an existing C0a checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  c1)
    CONFIG="config_moving_cube_c1"
    DEFAULT_NAME="moving_cube_c1_4096_no_rtx"
    ;;
  c2)
    CONFIG="config_moving_cube_c2"
    DEFAULT_NAME="moving_cube_c2_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c2 requires an existing C1 checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  c3h)
    CONFIG="config_moving_cube_c3h"
    DEFAULT_NAME="moving_cube_c3h_4096_no_rtx"
    if [[ $# -lt 1 || ! -f "$1" ]]; then
      echo "c3h requires an existing C2 checkpoint as its second argument" >&2
      exit 2
    fi
    INIT_ARGS=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  *)
    echo "unknown stage '$STAGE'; expected c0a, c0s, c0k, c0h, c0i, c0m, c0n, c0o, c0t, c0u, c0v, c0w, c0x, c0b, c1, c2, or c3h" >&2
    exit 2
    ;;
esac

export FOOD_ROBOT_ARTIFACTS="$REPO_ROOT/artifacts/moving_cube_static_bowl"
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

exec "$REPO_ROOT/.venv/bin/python" -X faulthandler pipeline/0_state_teacher/train.py \
  --config-name "$CONFIG" \
  env.num_envs=4096 \
  run.name="$DEFAULT_NAME" \
  worker.render=false \
  logger.backend=null \
  "${INIT_ARGS[@]}" \
  "$@"
