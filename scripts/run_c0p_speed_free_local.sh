#!/usr/bin/env bash
# Distil the selected C0m teacher into the GAP-style speed-free C0p policy.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TEACHER="${1:-artifacts/moving_cube_static_bowl/teachers/moving_cube_c0m_temporal_geometry_cont_6m_to12m/checkpoints/ppo_teacher_8060928.pt}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ ! -f "$TEACHER" ]]; then
  echo "C0m teacher checkpoint not found: $TEACHER" >&2
  exit 2
fi

export FOOD_ROBOT_ARTIFACTS="$REPO_ROOT/artifacts/moving_cube_static_bowl"
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

exec "$REPO_ROOT/.venv/bin/python" -X faulthandler pipeline/0_state_teacher/distill_speed_free.py \
  env.num_envs=4096 \
  "distill.teacher_checkpoint=$(realpath "$TEACHER")" \
  run.name=moving_cube_c0p_gap_speed_free_distill_4096_no_rtx \
  "$@"
