#!/usr/bin/env bash
# Dual-camera RGB-D data collection for the two frozen JAKA state teachers.
# Run inside the pinned project environment/container on the high-driver server.
#
# Usage:
#   scripts/run_jaka_visual_collection_server.sh moving_sphere_static_bowl smoke
#   scripts/run_jaka_visual_collection_server.sh static_sphere_moving_bowl pilot seed=1
# Set JAKA_VISUAL_DRY_RUN=1 to validate provenance and print the command only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TASK="${1:-}"
PROFILE="${2:-smoke}"
if [[ -z "$TASK" ]]; then
  echo "usage: $0 {moving_sphere_static_bowl|static_sphere_moving_bowl} {smoke|pilot|full} [collect.py overrides ...]" >&2
  exit 2
fi
shift
if [[ $# -gt 0 ]]; then
  shift
fi

case "$PROFILE" in
  smoke) FRAMES=4096; NUM_ENVS=32; ROLLOUT_STEPS=8 ;;
  pilot) FRAMES=100000; NUM_ENVS=128; ROLLOUT_STEPS=8 ;;
  full) FRAMES=1000000; NUM_ENVS=256; ROLLOUT_STEPS=8 ;;
  *) echo "unknown profile: $PROFILE" >&2; exit 2 ;;
esac

case "$TASK" in
  moving_sphere_static_bowl)
    ACTOR_CHECKPOINT_DEFAULT="$REPO_ROOT/artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j0_s1_probe500k/checkpoints/ppo_teacher_final.pt"
    ENV_CHECKPOINT_DEFAULT="$REPO_ROOT/artifacts/jaka_zu5_ag95_sphere/teachers/jaka_zu5_ag95_sphere_j1_place_006_008_1024_no_rtx/checkpoints/ppo_teacher_final.pt"
    ARTIFACT_ROOT="$REPO_ROOT/artifacts/jaka_zu5_ag95_sphere"
    ACTOR_SHA_DEFAULT="14b1d8af7ecaa46e4a77c48434ca07d43e7d2f3e73672832af663ece2485ea7b"
    ENV_SHA_DEFAULT="a945612cfb694adc424b7264635c20a34279e89e08c9aa20dc93c79c47450759"
    ;;
  static_sphere_moving_bowl)
    ACTOR_CHECKPOINT_DEFAULT="$REPO_ROOT/artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/teachers/jaka_static_sphere_moving_bowl_m0_final_geometry_smoke16/checkpoints/ppo_teacher_final.pt"
    ENV_CHECKPOINT_DEFAULT="$ACTOR_CHECKPOINT_DEFAULT"
    ARTIFACT_ROOT="$REPO_ROOT/artifacts/jaka_zu5_ag95_static_sphere_moving_bowl"
    ACTOR_SHA_DEFAULT="d522d818dccdb7a073708b75ccb44fb7bc9b98a5780e10f057b5faba3b01555b"
    ENV_SHA_DEFAULT="$ACTOR_SHA_DEFAULT"
    ;;
  *) echo "unknown JAKA visual task: $TASK" >&2; exit 2 ;;
esac

ACTOR_CHECKPOINT="${JAKA_VISUAL_ACTOR_CHECKPOINT:-$ACTOR_CHECKPOINT_DEFAULT}"
ENV_CHECKPOINT="${JAKA_VISUAL_ENV_CHECKPOINT:-$ENV_CHECKPOINT_DEFAULT}"
for required in "$ACTOR_CHECKPOINT" "$ENV_CHECKPOINT"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required checkpoint: $required" >&2
    exit 2
  fi
done

ACTOR_SHA_EXPECTED="${JAKA_VISUAL_ACTOR_SHA256:-$ACTOR_SHA_DEFAULT}"
ENV_SHA_EXPECTED="${JAKA_VISUAL_ENV_SHA256:-$ENV_SHA_DEFAULT}"
ACTOR_SHA_ACTUAL="$(sha256sum "$ACTOR_CHECKPOINT" | awk '{print $1}')"
ENV_SHA_ACTUAL="$(sha256sum "$ENV_CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTOR_SHA_ACTUAL" != "$ACTOR_SHA_EXPECTED" ]]; then
  echo "actor checkpoint SHA-256 mismatch: expected $ACTOR_SHA_EXPECTED, got $ACTOR_SHA_ACTUAL" >&2
  exit 2
fi
if [[ "$ENV_SHA_ACTUAL" != "$ENV_SHA_EXPECTED" ]]; then
  echo "environment checkpoint SHA-256 mismatch: expected $ENV_SHA_EXPECTED, got $ENV_SHA_ACTUAL" >&2
  exit 2
fi

if [[ -n "${PICKPLACE_PYTHON:-}" ]]; then
  PYTHON_BIN="$PICKPLACE_PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN=python
fi

SEED="${JAKA_VISUAL_SEED:-0}"
NAME="${JAKA_VISUAL_NAME:-jaka_${TASK}_${PROFILE}_rgbd_seed${SEED}}"
OUT="${JAKA_VISUAL_OUT:-$ARTIFACT_ROOT/g2_visual/shards/$NAME}"
export FOOD_ROBOT_ARTIFACTS="$ARTIFACT_ROOT"
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

COMMAND=(
  "$PYTHON_BIN" -X faulthandler pipeline/1_collect_data/collect.py
  "checkpoint=$ACTOR_CHECKPOINT"
  "env_checkpoint=$ENV_CHECKPOINT"
  "jaka_visual_task=$TASK"
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

"$PYTHON_BIN" scripts/inspect_g2_collection_plan.py \
  --checkpoint "$ACTOR_CHECKPOINT" \
  --env-checkpoint "$ENV_CHECKPOINT" \
  --jaka-visual-task "$TASK" \
  --profile "$PROFILE" \
  --image-size 84

if [[ "${JAKA_VISUAL_DRY_RUN:-0}" == "1" ]]; then
  printf 'JAKA_VISUAL_DRY_RUN '
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

"${COMMAND[@]}"
"$PYTHON_BIN" scripts/verify_g2_shard.py "$OUT" \
  --preview "$OUT/jaka_rgbd_preview.png"
