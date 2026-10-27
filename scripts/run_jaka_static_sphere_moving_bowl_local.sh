#!/usr/bin/env bash
# JAKA static-sphere -> moving-bowl curriculum.  On host robot, physical GPU0
# is forbidden and physical GPU1 is mapped to Isaac's logical cuda:0.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STAGE="${1:-m0}"
if [[ $# -gt 0 ]]; then shift; fi
case "$STAGE" in
  m0)
    CONFIG="config_jaka_static_sphere_moving_bowl_m0_speed010"
    DEFAULT_NAME="jaka_static_sphere_moving_bowl_m0_speed010_512_no_rtx"
    ;;
  *)
    echo "unknown stage '$STAGE'; expected m0" >&2
    exit 2
    ;;
esac

export FOOD_ROBOT_ARTIFACTS="$REPO_ROOT/artifacts/jaka_zu5_ag95_static_sphere_moving_bowl"
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
if [[ "$(hostname)" == "robot" ]]; then
  export PICKPLACE_PHYSICAL_GPU=1
fi

exec "$REPO_ROOT/.venv/bin/python" -X faulthandler pipeline/0_state_teacher/train.py \
  --config-name "$CONFIG" \
  env.num_envs="${JAKA_MOVING_BOWL_NUM_ENVS:-512}" \
  run.name="$DEFAULT_NAME" \
  worker.render=false \
  logger.backend=null \
  "$@"
