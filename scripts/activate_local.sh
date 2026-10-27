#!/usr/bin/env bash
# Activate the repository-local environment and keep every generated artifact
# and download cache inside this checkout.

_pickplace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$_pickplace_root/.venv/bin/activate" ]]; then
  echo "Missing $_pickplace_root/.venv; run 'bash scripts/setup_env.sh' first." >&2
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$_pickplace_root/.venv/bin/activate"

export XDG_CACHE_HOME="$_pickplace_root/.cache"
export UV_CACHE_DIR="$_pickplace_root/.cache/uv"
export PIP_CACHE_DIR="$_pickplace_root/.cache/pip"
export HF_HOME="$_pickplace_root/.cache/huggingface"
export FOOD_ROBOT_ARTIFACTS="$_pickplace_root/artifacts"
export FOOD_ROBOT_CACHE="$_pickplace_root/.cache/food_robot"

mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" \
  "$FOOD_ROBOT_ARTIFACTS" "$FOOD_ROBOT_CACHE" \
  "$_pickplace_root/datasets" "$_pickplace_root/weights"

echo "pickplace environment: $_pickplace_root/.venv"
echo "artifacts: $FOOD_ROBOT_ARTIFACTS"
echo "cache: $XDG_CACHE_HOME"

unset _pickplace_root
