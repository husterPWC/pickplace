#!/usr/bin/env bash
# G2 dual-camera RGB-D collection using the frozen G1 teacher.
# Usage:
#   PICKPLACE_PHYSICAL_GPU=0 scripts/run_g2_collection_server.sh smoke
#   PICKPLACE_PHYSICAL_GPU=3 scripts/run_g2_collection_server.sh pilot seed=1
# Set G2_DRY_RUN=1 to print the resolved command without launching Isaac.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${1:-smoke}"
shift || true
case "$PROFILE" in
  smoke) FRAMES=4096; NUM_ENVS=32; ROLLOUT_STEPS=8 ;;
  pilot) FRAMES=100000; NUM_ENVS=128; ROLLOUT_STEPS=8 ;;
  full) FRAMES=1000000; NUM_ENVS=256; ROLLOUT_STEPS=8 ;;
  *) echo "usage: $0 {smoke|pilot|full} [collect.py overrides ...]" >&2; exit 2 ;;
esac

CHECKPOINT="${G2_TEACHER_CHECKPOINT:-$REPO_ROOT/artifacts/moving_sphere_static_bowl/teachers/moving_sphere_g1_place_008_1024_no_rtx/checkpoints/ppo_teacher_262144.pt}"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing frozen G1 teacher: $CHECKPOINT" >&2
  exit 2
fi

if [[ -z "${PICKPLACE_PHYSICAL_GPU:-}" ]]; then
  if [[ "$(hostname)" == "robot" ]]; then
    export PICKPLACE_PHYSICAL_GPU=1
  else
    echo "set PICKPLACE_PHYSICAL_GPU explicitly on the multi-GPU server" >&2
    exit 2
  fi
fi
if [[ "$(hostname)" == "robot" && "$PICKPLACE_PHYSICAL_GPU" == "0" ]]; then
  echo "refusing physical GPU0 on host robot: cooling fault" >&2
  exit 2
fi

SEED="${G2_SEED:-0}"
NAME="${G2_NAME:-g2_${PROFILE}_rgbd_seed${SEED}}"
OUT="${G2_OUT:-$REPO_ROOT/artifacts/moving_sphere_static_bowl/g2_visual/shards/$NAME}"
export FOOD_ROBOT_ARTIFACTS="$REPO_ROOT/artifacts/moving_sphere_static_bowl"
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

COMMAND=(
  "$REPO_ROOT/.venv/bin/python" -X faulthandler pipeline/1_collect_data/collect.py
  "checkpoint=$CHECKPOINT"
  "out=$OUT"
  "name=$NAME"
  "frames=$FRAMES"
  "num_envs=$NUM_ENVS"
  "rollout_steps=$ROLLOUT_STEPS"
  "image_size=84"
  "seed=$SEED"
  "include_depth=true"
  "store_teacher_inputs=true"
  "$@"
)

"$REPO_ROOT/.venv/bin/python" scripts/inspect_g2_collection_plan.py \
  --checkpoint "$CHECKPOINT" --profile "$PROFILE" --image-size 84
if [[ "${G2_DRY_RUN:-0}" == "1" ]]; then
  printf 'G2_DRY_RUN '
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
"${COMMAND[@]}"
"$REPO_ROOT/.venv/bin/python" scripts/verify_g2_shard.py "$OUT" \
  --preview "$OUT/g2_rgbd_preview.png"
