#!/usr/bin/env bash
# Independent GAP-style SAC curriculum. Historical sphere S0 artifacts are untouched.
# Usage:
#   scripts/run_sphere_gap_curriculum_local.sh s0a [Hydra overrides ...]
#   scripts/run_sphere_gap_curriculum_local.sh s0b <s0a-best.pt> [Hydra overrides ...]
#   scripts/run_sphere_gap_curriculum_local.sh s0c <s0b-best.pt> [Hydra overrides ...]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ $# -ge 1 ]] || { echo "usage: $0 {s0a|s0b|s0c} [checkpoint] [overrides...]" >&2; exit 2; }
STAGE="$1"
shift
case "$STAGE" in
  s0a) CONFIG=config_sphere_gap_s0a; NAME=sphere_gap_s0a_sac_64_no_rtx; INIT=() ;;
  s0b|s0c)
    CONFIG="config_sphere_gap_${STAGE}"
    NAME="sphere_gap_${STAGE}_sac_64_no_rtx"
    [[ $# -ge 1 && -f "$1" ]] || { echo "$STAGE requires the preceding selected checkpoint" >&2; exit 2; }
    INIT=("init_checkpoint=$(realpath "$1")")
    shift
    ;;
  *) echo "unknown stage '$STAGE'" >&2; exit 2 ;;
esac

export FOOD_ROBOT_ARTIFACTS="$REPO_ROOT/artifacts/moving_sphere_static_bowl"
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

exec "$REPO_ROOT/.venv/bin/python" -X faulthandler pipeline/0_state_teacher/sac_train.py \
  --config-name "$CONFIG" run.name="$NAME" "${INIT[@]}" "$@"
