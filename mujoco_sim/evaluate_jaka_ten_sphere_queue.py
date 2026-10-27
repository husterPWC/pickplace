#!/usr/bin/env python3
"""Evaluate ten already-visible moving balls with one frozen JAKA policy."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from mujoco_sim.env import POLICY_DT, ROOT
from mujoco_sim.evaluate_jaka_ten_sphere_sequence import _sphere_snapshot
from mujoco_sim.jaka_moving_sphere import DEFAULT_CHECKPOINT, JakaMovingSphereMuJoCo
from mujoco_sim.moving_sphere import GraspPointGRUActor
from mujoco_sim.policy_runner import parse_cli


DEFAULT_REPORT = ROOT / (
    "artifacts/jaka_zu5_ag95_sphere/mujoco/"
    "ten_sphere_visible_queue_speed008_checkpoint_closed_loop.json"
)


def _settle_world(env: JakaMovingSphereMuJoCo, seconds: float) -> None:
    steps = int(round(seconds / env.model.opt.timestep))
    for _ in range(steps):
        env._apply_ik()
        env.data.ctrl[env.finger_act] = 0.0
        env._set_pallet_targets(float(env.data.time) + env.model.opt.timestep)
        mujoco.mj_step(env.model, env.data)
    mujoco.mj_forward(env.model, env.data)


def main() -> None:
    cli = parse_cli()
    checkpoint = Path(cli.get("checkpoint", DEFAULT_CHECKPOINT)).resolve()
    report_out = Path(cli.get("report_out", DEFAULT_REPORT)).resolve()
    seed = int(cli.get("seed", 8210))
    num_spheres = int(cli.get("num_spheres", 10))
    belt_speed = float(cli.get("belt_speed", 0.08))
    spacing = float(cli.get("spacing", 0.71))
    max_seconds_per_call = float(cli.get("max_seconds_per_call", 12.0))
    if num_spheres < 1:
        raise ValueError("num_spheres must be positive")

    actor = GraspPointGRUActor(checkpoint)
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
        sphere_queue_spacing=spacing,
    )
    observation = env.reset(seed)
    initial_positions = env.data.xpos[env.food_body_all].copy()
    initial_velocities = np.asarray(
        [env.data.qvel[dof : dof + 3].copy() for dof in env.food_dof_all]
    )
    all_present_at_start = bool(
        np.all(env.model.geom_contype[env.food_geom_all] != 0)
        and np.all(env.model.geom_conaffinity[env.food_geom_all] != 0)
    )
    all_moving_at_start = bool(np.all(initial_velocities[:, 0] > 0.95 * belt_speed))

    calls: list[dict[str, object]] = []
    all_calls_succeeded = True
    all_returns_succeeded = True
    for sphere_index in range(num_spheres):
        if sphere_index > 0:
            observation = env.prepare_sphere_call(sphere_index)
        invocation_position = env.data.xpos[env.food_body].copy()
        invocation_world_time = float(env.data.time)
        result = None
        phases: list[str] = []
        action_abs_sum = 0.0
        action_values = 0
        action_max_abs = 0.0
        nonzero_action_steps = 0
        for _ in range(int(max_seconds_per_call / POLICY_DT) + 1):
            network_action = actor.act(observation)
            action_abs_sum += float(np.abs(network_action).sum())
            action_values += int(network_action.size)
            action_max_abs = max(action_max_abs, float(np.abs(network_action).max()))
            nonzero_action_steps += int(np.any(np.abs(network_action) > 1.0e-8))
            result = env.step(network_action)
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
                "invocation_world_time": round(invocation_world_time, 4),
                "sphere_position_at_invocation": invocation_position.tolist(),
                "outcome": result.outcome,
                "success": call_success,
                "policy_seconds": round(env.elapsed, 3),
                "network_control_steps": action_values // 7,
                "network_nonzero_action_steps": nonzero_action_steps,
                "network_mean_abs_action": action_abs_sum / max(action_values, 1),
                "network_max_abs_action": action_max_abs,
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
    belt_geom = env.model.geom("belt_visual").id
    belt_center = float(env.model.geom_pos[belt_geom, 0])
    belt_half_length = float(env.model.geom_size[belt_geom, 0])
    report = {
        "engine": "mujoco",
        "physics_closed_loop": True,
        "robot": "jaka_zu5_ag95",
        "task": "ten_visible_balls_single_frozen_policy_queue",
        "controller": "frozen_checkpoint_residual_plus_geometry_baseline",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "network_called_every_control_step": True,
        "zero_residual_baseline": False,
        "translation_residual_limit": 0.05,
        "rotation_residual_limit": 0.025,
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
        "sphere_spacing_m": spacing,
        "belt_upstream_x": belt_center - belt_half_length,
        "belt_downstream_x": belt_center + belt_half_length,
        "belt_length_m": 2.0 * belt_half_length,
        "all_balls_present_at_start": all_present_at_start,
        "all_balls_moving_simultaneously_at_start": all_moving_at_start,
        "spawned_between_calls": False,
        "world_reset_between_calls": False,
        "bowl_reset_between_calls": False,
        "placed_balls_removed": False,
        "grasp_assist": False,
        "initial_sphere_positions": initial_positions.tolist(),
        "initial_sphere_velocities": initial_velocities.tolist(),
        "calls": calls,
        "final_spheres": sphere_records,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("JAKA_TEN_SPHERE_VISIBLE_QUEUE " + json.dumps(report), flush=True)
    env.close()


if __name__ == "__main__":
    main()
