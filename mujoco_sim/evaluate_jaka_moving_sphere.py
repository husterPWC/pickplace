#!/usr/bin/env python3
"""Deterministic closed-loop MuJoCo probe for the JAKA J0 policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mujoco_sim.env import POLICY_DT
from mujoco_sim.jaka_moving_sphere import DEFAULT_CHECKPOINT, JakaMovingSphereMuJoCo
from mujoco_sim.moving_sphere import GraspPointGRUActor
from mujoco_sim.policy_runner import as_bool, parse_cli


def main() -> None:
    cli = parse_cli()
    checkpoint = Path(cli.get("checkpoint", DEFAULT_CHECKPOINT)).resolve()
    seed = int(cli.get("seed", 31))
    belt_speed = float(cli.get("belt_speed", 0.04))
    max_seconds = float(cli.get("max_seconds", 45.0))
    grasp_assist = as_bool(cli.get("grasp_assist", False))
    controller_profile = str(cli.get("controller_profile", "j0_s1"))
    pick_place = as_bool(cli.get("pick_place", controller_profile == "j1"))
    nominal_bowl_pos = (
        float(cli.get("bowl_x", 0.45)),
        float(cli.get("bowl_y", 0.0)),
        float(cli.get("bowl_z", 0.0)),
    )
    episodes = int(cli.get("episodes", 1))
    randomize = as_bool(cli.get("randomize", False))
    if episodes < 1:
        raise ValueError("episodes must be >= 1")
    actor = GraspPointGRUActor(checkpoint)
    rng = np.random.default_rng(seed)
    episode_records = []
    for episode in range(episodes):
        if randomize:
            episode_speed = float(rng.uniform(0.075, 0.080))
            sphere_x = float(rng.uniform(0.19, 0.21))
            sphere_y = float(rng.uniform(0.29, 0.31))
            bowl_pos = (
                float(nominal_bowl_pos[0] + rng.uniform(-0.01, 0.01)),
                float(nominal_bowl_pos[1] + rng.uniform(-0.01, 0.01)),
                nominal_bowl_pos[2],
            )
        else:
            episode_speed = belt_speed
            sphere_x, sphere_y = 0.20, 0.30
            bowl_pos = nominal_bowl_pos
        env = JakaMovingSphereMuJoCo(
            seed=seed + episode,
            belt_speed=episode_speed,
            grasp_assist=grasp_assist,
            max_seconds=max_seconds,
            sphere_start_x=sphere_x,
            sphere_start_y=sphere_y,
            controller_profile=controller_profile,
            pick_place=pick_place,
            destination_bowl_pos=bowl_pos,
        )
        observation = env.reset(seed + episode)
        initial_tcp = observation["ee_pos"].copy()
        result = None
        phases: list[str] = []
        for _ in range(int(max_seconds / POLICY_DT) + 1):
            result = env.step(actor.act(observation))
            observation = result.observation
            phase = str(result.info["option_phase"])
            if not phases or phases[-1] != phase:
                phases.append(phase)
            if result.done:
                break
        if result is None:
            raise RuntimeError("rollout produced no steps")
        episode_records.append({
            "episode": episode,
            "seed": seed + episode,
            "belt_speed": episode_speed,
            "sphere_start": [sphere_x, sphere_y],
            "destination_bowl_pos": list(bowl_pos),
            "outcome": result.outcome,
            "phases": phases,
            "episode_seconds": round(env.elapsed, 3),
            "initial_tcp": [round(float(value), 6) for value in initial_tcp],
            "final_info": result.info,
        })
        env.close()
    expected = "strict_place_success" if pick_place else "strict_lift_success"
    successes = sum(record["outcome"] == expected for record in episode_records)
    last = episode_records[-1]
    report = {
        "engine": "mujoco",
        "physics_closed_loop": True,
        "robot": "jaka_zu5_ag95",
        "checkpoint": str(checkpoint),
        "seed": seed,
        "belt_speed": belt_speed,
        "grasp_assist": grasp_assist,
        "controller_profile": controller_profile,
        "pick_place": pick_place,
        "destination_bowl_pos": list(nominal_bowl_pos),
        "episodes": episodes,
        "randomize": randomize,
        "successes": successes,
        "success_rate": successes / episodes,
        "expected_outcome": expected,
        "outcome": last["outcome"],
        "phases": last["phases"],
        "episode_seconds": last["episode_seconds"],
        "initial_tcp": last["initial_tcp"],
        "final_info": last["final_info"],
        "episode_records": episode_records,
    }
    if "report_out" in cli:
        report_out = Path(cli["report_out"]).resolve()
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("JAKA_MUJOCO_EVAL " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
