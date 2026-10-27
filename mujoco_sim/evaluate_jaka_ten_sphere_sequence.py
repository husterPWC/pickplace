#!/usr/bin/env python3
"""Call the frozen JAKA single-ball policy ten times in one MuJoCo world."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from mujoco_sim.env import POLICY_DT, ROOT
from mujoco_sim.jaka_moving_sphere import DEFAULT_CHECKPOINT, JakaMovingSphereMuJoCo
from mujoco_sim.moving_sphere import GraspPointGRUActor
from mujoco_sim.policy_runner import as_bool, parse_cli


DEFAULT_REPORT = ROOT / (
    "artifacts/jaka_zu5_ag95_sphere/mujoco/"
    "ten_sphere_repeated_policy_speed008.json"
)


def _sphere_snapshot(env: JakaMovingSphereMuJoCo) -> list[dict[str, object]]:
    bowl = env.destination_bowl_pos
    records = []
    for index, (body, dof) in enumerate(zip(env.food_body_all, env.food_dof_all)):
        position = env.data.xpos[body].copy()
        velocity = env.data.qvel[dof : dof + 3].copy()
        relative = position - bowl
        retained = bool(
            np.linalg.norm(relative[:2]) < 0.065
            and 0.005 < relative[2] < 0.11
            and np.linalg.norm(velocity) < 0.05
        )
        records.append(
            {
                "sphere": index,
                "position": position.tolist(),
                "velocity": velocity.tolist(),
                "bowl_xy_distance": float(np.linalg.norm(relative[:2])),
                "bowl_relative_z": float(relative[2]),
                "retained_and_settled": retained,
            }
        )
    return records


def _settle_world(env: JakaMovingSphereMuJoCo, seconds: float) -> None:
    steps = int(round(seconds / env.model.opt.timestep))
    for _ in range(steps):
        env._apply_ik()
        env.data.ctrl[env.finger_act] = 0.0
        mujoco.mj_step(env.model, env.data)
    mujoco.mj_forward(env.model, env.data)


def main() -> None:
    cli = parse_cli()
    checkpoint = Path(cli.get("checkpoint", DEFAULT_CHECKPOINT)).resolve()
    report_out = Path(cli.get("report_out", DEFAULT_REPORT)).resolve()
    seed = int(cli.get("seed", 8200))
    num_spheres = int(cli.get("num_spheres", 10))
    belt_speed = float(cli.get("belt_speed", 0.08))
    max_seconds_per_call = float(cli.get("max_seconds_per_call", 12.0))
    randomize = as_bool(cli.get("randomize", False))
    if num_spheres < 1:
        raise ValueError("num_spheres must be positive")

    actor = GraspPointGRUActor(checkpoint)
    rng = np.random.default_rng(seed)
    env = JakaMovingSphereMuJoCo(
        seed=seed,
        belt_speed=belt_speed,
        grasp_assist=False,
        max_seconds=max_seconds_per_call,
        sphere_start_x=0.20,
        sphere_start_y=0.30,
        controller_profile="j1",
        pick_place=True,
        destination_bowl_pos=(0.45, 0.0, 0.0),
        num_spheres=num_spheres,
    )

    observation = env.reset(seed)
    calls: list[dict[str, object]] = []
    all_calls_succeeded = True
    all_returns_succeeded = True
    for sphere_index in range(num_spheres):
        start_x = 0.20 + (float(rng.uniform(-0.008, 0.008)) if randomize else 0.0)
        start_y = 0.30 + (float(rng.uniform(-0.008, 0.008)) if randomize else 0.0)
        if sphere_index > 0 or randomize:
            observation = env.prepare_sphere_call(
                sphere_index,
                sphere_start_x=start_x,
                sphere_start_y=start_y,
            )

        result = None
        phases: list[str] = []
        for _ in range(int(max_seconds_per_call / POLICY_DT) + 1):
            result = env.step(actor.act(observation))
            observation = result.observation
            phase = str(result.info["option_phase"])
            if not phases or phases[-1] != phase:
                phases.append(phase)
            if result.done:
                break
        if result is None:
            raise RuntimeError("policy invocation produced no steps")
        call_success = result.outcome == "strict_place_success"
        all_calls_succeeded &= call_success
        return_result = env.return_to_intercept_ready()
        all_returns_succeeded &= bool(return_result["success"])
        calls.append(
            {
                "call": sphere_index + 1,
                "sphere": sphere_index,
                "sphere_start": [start_x, start_y],
                "outcome": result.outcome,
                "success": call_success,
                "policy_seconds": round(env.elapsed, 3),
                "phases": phases,
                "return_to_ready": return_result,
                "final_info": result.info,
            }
        )
        if not call_success:
            break

    _settle_world(env, 1.0)
    sphere_records = _sphere_snapshot(env)
    retained = sum(bool(record["retained_and_settled"]) for record in sphere_records)
    completed_calls = len(calls)
    strict_success = bool(
        completed_calls == num_spheres
        and all_calls_succeeded
        and all_returns_succeeded
        and retained == num_spheres
    )
    report = {
        "engine": "mujoco",
        "physics_closed_loop": True,
        "robot": "jaka_zu5_ag95",
        "task": "repeat_frozen_single_ball_policy",
        "checkpoint": str(checkpoint),
        "same_actor_reused": True,
        "actor_reload_count": 1,
        "policy_calls_requested": num_spheres,
        "policy_calls_completed": completed_calls,
        "successful_policy_calls": sum(bool(call["success"]) for call in calls),
        "return_to_ready_successes": sum(
            bool(call["return_to_ready"]["success"]) for call in calls
        ),
        "balls_retained_and_settled": retained,
        "strict_sequence_success": strict_success,
        "belt_speed_mps": belt_speed,
        "randomize_sphere_start": randomize,
        "grasp_assist": False,
        "world_reset_between_calls": False,
        "bowl_reset_between_calls": False,
        "placed_balls_removed": False,
        "calls": calls,
        "final_spheres": sphere_records,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("JAKA_TEN_SPHERE_SEQUENCE " + json.dumps(report), flush=True)
    env.close()


if __name__ == "__main__":
    main()
