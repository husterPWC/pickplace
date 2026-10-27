#!/usr/bin/env python3
"""Strict CPU-only MuJoCo evaluation for JAKA static sphere -> moving bowl."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mujoco_sim.env import POLICY_DT, ROOT
from mujoco_sim.jaka_moving_sphere import JakaMovingSphereMuJoCo
from mujoco_sim.policy_runner import as_bool, parse_cli


DEFAULT_REPORT = ROOT / (
    "artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/mujoco/"
    "strict_eval_speed010.json"
)


def main() -> None:
    cli = parse_cli()
    seed = int(cli.get("seed", 7100))
    episodes = int(cli.get("episodes", 20))
    bowl_speed = float(cli.get("bowl_speed", 0.10))
    max_seconds = float(cli.get("max_seconds", 12.0))
    randomize = as_bool(cli.get("randomize", True))
    report_out = Path(cli.get("report_out", DEFAULT_REPORT)).resolve()
    if episodes < 1:
        raise ValueError("episodes must be >= 1")

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for episode in range(episodes):
        sphere_x = 0.45
        sphere_y = 0.00
        if randomize:
            sphere_x += float(rng.uniform(-0.008, 0.008))
            sphere_y += float(rng.uniform(-0.008, 0.008))
        env = JakaMovingSphereMuJoCo(
            seed=seed + episode,
            belt_speed=bowl_speed,
            grasp_assist=False,
            max_seconds=max_seconds,
            sphere_start_x=sphere_x,
            sphere_start_y=sphere_y,
            controller_profile="static_pick",
            pick_place=True,
            task_variant="static_sphere_moving_bowl",
        )
        env.reset(seed + episode)
        result = None
        phases: list[str] = []
        for _ in range(int(max_seconds / POLICY_DT) + 1):
            result = env.step(np.zeros(7, dtype=np.float32))
            phase = str(result.info["option_phase"])
            if not phases or phases[-1] != phase:
                phases.append(phase)
            if result.done:
                break
        if result is None:
            raise RuntimeError("rollout produced no steps")
        records.append(
            {
                "episode": episode,
                "seed": seed + episode,
                "sphere_start": [sphere_x, sphere_y],
                "outcome": result.outcome,
                "phases": phases,
                "episode_seconds": round(env.elapsed, 3),
                "final_info": result.info,
            }
        )
        env.close()

    expected = "strict_place_home_success"
    successes = sum(record["outcome"] == expected for record in records)
    report = {
        "engine": "mujoco",
        "physics_closed_loop": True,
        "robot": "jaka_zu5_ag95",
        "task": "static_sphere_to_moving_bowl",
        "controller": "geometry_baseline_plus_zero_residual",
        "actor_speed_input": False,
        "grasp_assist": False,
        "bowl_speed_mps": bowl_speed,
        "episodes": episodes,
        "randomize_sphere_start": randomize,
        "successes": successes,
        "success_rate": successes / episodes,
        "expected_outcome": expected,
        "records": records,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("JAKA_STATIC_TO_MOVING_EVAL " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
