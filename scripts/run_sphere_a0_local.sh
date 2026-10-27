#!/usr/bin/env bash
# Run the four-environment A0 learnability gate on the local 3090.
# All caches, logs and checkpoints remain inside this repository.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export FOOD_ROBOT_ARTIFACTS="$REPO_ROOT/artifacts/moving_sphere_static_bowl"
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

exec "$REPO_ROOT/.venv/bin/python" -X faulthandler pipeline/0_state_teacher/train.py \
  --config-name config_sphere_a0_minimal \
  "$@"
