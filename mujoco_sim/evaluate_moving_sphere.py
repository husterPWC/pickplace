#!/usr/bin/env python3
"""Headless closed-loop acceptance for the moving-sphere MuJoCo bridge.

The evaluator deliberately uses CPU MuJoCo and never exposes the physical
contact diagnostics to the actor.  It supports both a frozen operating point
and explicit G1 position randomisation for repeatability checks.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from mujoco_sim.moving_sphere import (
    BELT_Y,
    POLICY_DT,
    ROOT,
    FrankaMovingSphereMuJoCo,
    GraspPointGRUActor,
)
from mujoco_sim.policy_runner import as_bool, parse_cli


DEFAULT_CHECKPOINT = ROOT / (
    "artifacts/moving_sphere_static_bowl/teachers/"
    "moving_sphere_g0_s2c_align_follow_speed_008_1024_no_rtx/"
    "checkpoints/ppo_teacher_507904.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "artifacts/moving_sphere_static_bowl/mujoco/"
    "g0_s2c_best507904_speed008_pure_contact_20ep.json"
)


def optional_float(value: str | None) -> float | None:
    return None if value in (None, "none", "null") else float(value)


def main() -> None:
    cli = parse_cli()
    checkpoint = Path(cli.get("checkpoint", DEFAULT_CHECKPOINT)).resolve()
    output = Path(cli.get("out", DEFAULT_OUTPUT)).resolve()
    episodes = int(cli.get("episodes", 20))
    seed = int(cli.get("seed", 0))
    # This acceptance entrypoint is tied to the frozen G0-S2c 0.08 m/s
    # operating point, not moving_sphere.py's historical 0.01 m/s default.
    belt_speed = float(cli.get("belt_speed", 0.08))
    grasp_assist = as_bool(cli.get("grasp_assist", False))
    pick_place = as_bool(cli.get("pick_place", False))
    randomize_position = as_bool(cli.get("randomize_position", False))
    settings = {
        "belt_speed": belt_speed,
        "grasp_assist": grasp_assist,
        "controller": cli.get("controller", "residual_baseline"),
        "intercept_lead_distance": float(cli.get("intercept_lead_distance", 0.16)),
        "intercept_align_lead_distance": float(cli.get("intercept_align_lead_distance", 0.06)),
        "pre_descent_height": optional_float(cli.get("pre_descent_height", "0.02")),
        "lift_steps": int(cli.get("lift_steps", 60)),
        "lift_height": float(cli.get("lift_height", 0.05)),
        "lift_xy_gain": optional_float(cli.get("lift_xy_gain", "0.01")),
        "max_lift_xy_step": optional_float(cli.get("max_lift_xy_step", "0.002")),
        "preserve_lift_relative_xy": as_bool(cli.get("preserve_lift_relative_xy", True)),
        "target_xy_feedforward_gain": float(cli.get("target_xy_feedforward_gain", 4.0)),
        "max_target_xy_feedforward_step": optional_float(
            cli.get("max_target_xy_feedforward_step", "0.008")
        ),
        "align_target_xy_feedforward": as_bool(cli.get("align_target_xy_feedforward", True)),
        "close_requires_raw_alignment": as_bool(cli.get("close_requires_raw_alignment", True)),
        "max_close_hold_xy_step": optional_float(cli.get("max_close_hold_xy_step", "0.008")),
        "latch_lift_xy_feedforward": as_bool(cli.get("latch_lift_xy_feedforward", True)),
        "lift_geometry_relatch_height": optional_float(
            cli.get("lift_geometry_relatch_height", "0.012")
        ),
        "sphere_start_x": float(cli.get("sphere_start_x", -0.18)),
        "belt_upstream_x": optional_float(cli.get("belt_upstream_x", "-0.35")),
        "pallet_start_x": optional_float(cli.get("pallet_start_x", "-0.18")),
        "lift_timeout_steps": int(cli.get("lift_timeout_steps", 1200)),
        "max_seconds": float(cli.get("max_seconds", 60.0)),
        "pick_place": pick_place,
        "translation_residual_limit": float(
            cli.get("translation_residual_limit", 0.05 if pick_place else 0.10)
        ),
        "rotation_residual_limit": float(
            cli.get("rotation_residual_limit", 0.025 if pick_place else 0.05)
        ),
    }
    if grasp_assist:
        raise ValueError("pure-contact acceptance requires grasp_assist=false")

    actor = GraspPointGRUActor(checkpoint)
    records: list[dict[str, object]] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        episode_settings = dict(settings)
        if randomize_position:
            rng = np.random.default_rng(episode_seed)
            pallet_start_x = float(rng.uniform(-0.20, -0.16))
            sphere_start_x = pallet_start_x + float(rng.uniform(-0.01, 0.01))
            sphere_start_y = BELT_Y + float(rng.uniform(-0.01, 0.01))
            bowl_x = float(rng.uniform(0.44, 0.46))
            bowl_y = float(rng.uniform(-0.01, 0.01))
            episode_settings.update(
                pallet_start_x=pallet_start_x,
                sphere_start_x=sphere_start_x,
                sphere_start_y=sphere_start_y,
                destination_bowl_pos=(bowl_x, bowl_y, 0.0),
            )
        else:
            episode_settings.update(
                sphere_start_y=float(cli.get("sphere_start_y", BELT_Y)),
                destination_bowl_pos=(
                    float(cli.get("bowl_x", 0.45)),
                    float(cli.get("bowl_y", 0.0)),
                    0.0,
                ),
            )
        env = FrankaMovingSphereMuJoCo(seed=episode_seed, **episode_settings)
        try:
            observation = env.reset(episode_seed)
            steps = 0
            while True:
                result = env.step(actor.act(observation))
                observation = result.observation
                steps += 1
                if result.done:
                    record = {
                        "episode": episode,
                        "seed": episode_seed,
                        "outcome": result.outcome,
                        "steps": steps,
                        "seconds": round(steps * POLICY_DT, 3),
                        "min_translation": env.diagnostics.min_translation,
                        "min_rotation": env.diagnostics.min_rotation,
                        "initial_pallet_x": env.pallet_start_x,
                        "initial_sphere_x": env.sphere_start_x,
                        "initial_sphere_y": env.sphere_start_y,
                        "destination_bowl_x": float(env.destination_bowl_pos[0]),
                        "destination_bowl_y": float(env.destination_bowl_pos[1]),
                        **result.info,
                    }
                    records.append(record)
                    print("MUJOCO_SPHERE_EPISODE " + json.dumps(record), flush=True)
                    break
        finally:
            env.close()

    counts = Counter(str(record["outcome"]) for record in records)
    success_outcome = "strict_place_success" if pick_place else "strict_lift_success"
    successes = counts.get(success_outcome, 0)
    report = {
        "checkpoint": str(checkpoint),
        "engine": "mujoco",
        "physics_closed_loop": True,
        "gpu_used": False,
        "policy_input_has_explicit_velocity": False,
        "policy_input_has_contact_or_force": False,
        "pick_place": pick_place,
        "randomize_position": randomize_position,
        "success_outcome": success_outcome,
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / max(1, len(records)),
        "outcomes": dict(counts),
        "all_ever_dual_contact": all(bool(record["ever_dual_contact"]) for record in records),
        "all_successes_had_dual_contact": all(
            bool(record["ever_dual_contact"])
            for record in records
            if record["outcome"] == success_outcome
        ),
        # For pick/place this is expected to be false after a successful open
        # release.  Keep it explicitly labelled as a final-state diagnostic.
        "all_final_physical_grasp_secured": all(
            bool(record["physical_grasp_secured"]) for record in records
        ),
        "settings": {"seed": seed, "policy_dt": POLICY_DT, **settings},
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "MUJOCO_SPHERE_ACCEPTANCE "
        + json.dumps({key: value for key, value in report.items() if key != "records"}),
        flush=True,
    )
    print(f"MUJOCO_SPHERE_ACCEPTANCE_DONE {output}", flush=True)


if __name__ == "__main__":
    main()
