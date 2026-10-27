#!/usr/bin/env python3
"""Evaluate the adapted closed-loop policy over conveyor speeds in parallel."""

from __future__ import annotations

import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from mujoco_sim.env import FrankaPickPlaceMuJoCo, ROOT
from mujoco_sim.policy_runner import load_actor, parse_cli


DEFAULT_CHECKPOINT = ROOT / "artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/mujoco/franka_v3c_belt_speed_sweep.json"
_actor = None


def initialize_worker(checkpoint: str) -> None:
    global _actor
    _actor = load_actor(Path(checkpoint))


def evaluate_one(task: tuple[float, int]) -> dict:
    speed, seed = task
    env = FrankaPickPlaceMuJoCo(
        seed=seed,
        randomize=True,
        grasp_assist=True,
        belt_speed=speed,
    )
    _actor.reset()
    observation = env.reset(seed)
    ever_grasped = False
    for step in range(600):
        result = env.step(_actor.act(observation))
        observation = result.observation
        ever_grasped |= bool(result.info["is_grasped"])
        if result.done:
            return {
                "belt_speed": speed,
                "seed": seed,
                "outcome": result.outcome,
                "steps": step + 1,
                "seconds": round((step + 1) * 0.02, 3),
                "ever_grasped": ever_grasped,
                "placed": bool(result.info["placed"]),
                "home_distance": float(result.info["home_distance"]),
            }
    raise RuntimeError(f"step limit reached for speed={speed}, seed={seed}")


def main() -> None:
    cli = parse_cli()
    checkpoint = Path(cli.get("checkpoint", DEFAULT_CHECKPOINT)).resolve()
    output = Path(cli.get("out", DEFAULT_OUTPUT)).resolve()
    speeds = [float(value) for value in cli.get("speeds", "0.04,0.06,0.08,0.10,0.12,0.14").split(",")]
    episodes = int(cli.get("episodes", 20))
    seed = int(cli.get("seed", 0))
    workers = int(cli.get("workers", min(4, os.cpu_count() or 1)))
    tasks = [(speed, seed + episode) for speed in speeds for episode in range(episodes)]

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(str(checkpoint),),
    ) as pool:
        records = list(pool.map(evaluate_one, tasks))

    summaries = []
    for speed in speeds:
        selected = [record for record in records if np.isclose(record["belt_speed"], speed)]
        counts = Counter(record["outcome"] for record in selected)
        durations = [record["seconds"] for record in selected]
        summary = {
            "belt_speed": speed,
            "episodes": len(selected),
            "successes": counts.get("success", 0),
            "success_rate": counts.get("success", 0) / len(selected),
            "outcomes": dict(counts),
            "mean_seconds": float(np.mean(durations)),
        }
        summaries.append(summary)
        print("SPEED_SWEEP " + json.dumps(summary), flush=True)

    payload = {
        "checkpoint": str(checkpoint),
        "seed": seed,
        "episodes_per_speed": episodes,
        "summaries": summaries,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"SPEED_SWEEP_DONE {output}", flush=True)


if __name__ == "__main__":
    main()
