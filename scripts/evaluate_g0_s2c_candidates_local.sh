#!/usr/bin/env bash
# Deterministically evaluate every G0-S2c short-run checkpoint at 0.08 m/s.
# The host robot's physical GPU0 is retired; only physical GPU1 is permitted.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMNI_KIT_ACCEPT_EULA=YES
export PICKPLACE_PHYSICAL_GPU=1
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export FOOD_ROBOT_CACHE="$REPO_ROOT/.cache/food_robot"

RUN_DIR="$REPO_ROOT/artifacts/moving_sphere_static_bowl/teachers/moving_sphere_g0_s2c_align_follow_speed_008_1024_no_rtx"
DIAG_DIR="$REPO_ROOT/artifacts/moving_sphere_static_bowl/diagnostics"
EVAL_NUM_ENVS="${G0_S2C_EVAL_NUM_ENVS:-64}"
mkdir -p "$DIAG_DIR"

if [[ $# -gt 0 ]]; then
  names=("$@")
else
  names=(ppo_teacher_262144 ppo_teacher_507904 ppo_teacher_753664 ppo_teacher_final)
fi

for name in "${names[@]}"; do
  checkpoint="$RUN_DIR/checkpoints/$name.pt"
  report="$DIAG_DIR/g0_s2c_${name}_eval${EVAL_NUM_ENVS}_speed008.json"
  log="$DIAG_DIR/g0_s2c_${name}_eval${EVAL_NUM_ENVS}_speed008.log"
  echo "EVALUATING $name"
  "$REPO_ROOT/.venv/bin/python" scripts/diagnose_sphere_early_intercept.py "$checkpoint" \
    --num-envs "$EVAL_NUM_ENVS" \
    --seed 0 \
    --speed 0.08 \
    --pallet-start -0.38 \
    --visual-upstream-extension 0.25 \
    --episode-completion-margin 10 \
    --lead 0.16 \
    --align-lead 0.06 \
    --pre-descent-height 0.02 \
    --lift-timeout-steps 1200 \
    --lift-xy-gain 0.01 \
    --max-lift-xy-step 0.002 \
    --target-xy-feedforward-gain 4.0 \
    --max-target-xy-feedforward-step 0.008 \
    --align-target-xy-feedforward \
    --close-requires-raw-alignment \
    --max-close-hold-xy-step 0.008 \
    --latch-lift-xy-feedforward \
    --lift-geometry-relatch-height 0.012 \
    --translation-residual-limit 0.10 \
    --rotation-residual-limit 0.05 \
    --trace-stride 50 \
    --output "$report" >"$log" 2>&1
  echo "RESULT $name"
  rg -n '"successes"|"outcomes"|"phase_reached"|"ever_dual_contact"|"ever_coupled"|"true_lift_m"' "$report" | head -12
done
