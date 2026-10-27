#!/usr/bin/env python3
"""Strict ten-static-sphere to ten-moving-bowl MuJoCo queue evaluation."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from mujoco_sim.env import POLICY_DT, ROOT
from mujoco_sim.jaka_moving_sphere import JakaMovingSphereMuJoCo
from mujoco_sim.moving_sphere import GraspPointGRUActor
from mujoco_sim.policy_runner import parse_cli


DEFAULT_REPORT = ROOT / (
    "artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/mujoco/"
    "ten_static_spheres_ten_visible_moving_bowls_speed010_checkpoint_closed_loop.json"
)
DEFAULT_CHECKPOINT = ROOT / (
    "artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/teachers/"
    "jaka_static_sphere_moving_bowl_m0_final_geometry_smoke16/checkpoints/"
    "ppo_teacher_final.pt"
)


def pair_snapshot(env: JakaMovingSphereMuJoCo) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, (sphere_body, sphere_dof, bowl_body) in enumerate(
        zip(env.food_body_all, env.food_dof_all, env.moving_bowl_body_all)
    ):
        sphere_position = env.data.xpos[sphere_body].copy()
        bowl_position = env.data.xpos[bowl_body].copy()
        relative = sphere_position - bowl_position
        sphere_velocity = env.data.qvel[sphere_dof : sphere_dof + 3].copy()
        bowl_velocity = env.data.cvel[bowl_body, 3:].copy()
        relative_speed = float(np.linalg.norm(sphere_velocity - bowl_velocity))
        retained = bool(
            np.linalg.norm(relative[:2]) < 0.065
            and 0.005 < relative[2] < 0.09
            and relative_speed < 0.05
        )
        records.append(
            {
                "pair": index,
                "sphere_position": sphere_position.tolist(),
                "bowl_position": bowl_position.tolist(),
                "sphere_to_bowl_xy": float(np.linalg.norm(relative[:2])),
                "sphere_to_bowl_z": float(relative[2]),
                "sphere_to_bowl_speed": relative_speed,
                "retained_and_settled": retained,
            }
        )
    return records


def settle_world(env: JakaMovingSphereMuJoCo, seconds: float) -> None:
    for _ in range(round(seconds / env.model.opt.timestep)):
        env._apply_ik()
        env.data.ctrl[env.finger_act] = 0.0
        env._set_pallet_targets(float(env.data.time) + env.model.opt.timestep)
        mujoco.mj_step(env.model, env.data)
    mujoco.mj_forward(env.model, env.data)


def main() -> None:
    cli = parse_cli()
    report_out = Path(cli.get("report_out", DEFAULT_REPORT)).resolve()
    checkpoint = Path(cli.get("checkpoint", DEFAULT_CHECKPOINT)).resolve()
    seed = int(cli.get("seed", 8300))
    num_items = int(cli.get("num_items", 10))
    bowl_speed = float(cli.get("bowl_speed", 0.10))
    spacing = float(cli.get("spacing", 0.75))
    max_seconds_per_call = float(cli.get("max_seconds_per_call", 12.0))

    actor = GraspPointGRUActor(checkpoint)
    env = JakaMovingSphereMuJoCo(
        seed=seed,
        belt_speed=bowl_speed,
        grasp_assist=False,
        max_seconds=max_seconds_per_call,
        controller_profile="static_pick",
        pick_place=True,
        task_variant="static_sphere_moving_bowl",
        num_spheres=num_items,
        moving_bowl_queue_spacing=spacing,
        translation_residual_limit=0.03,
        rotation_residual_limit=0.015,
    )
    observation = env.reset(seed)
    initial_sphere_positions = env.data.xpos[env.food_body_all].copy()
    initial_bowl_positions = env.data.xpos[env.moving_bowl_body_all].copy()
    initial_sphere_velocities = np.asarray(
        [env.data.qvel[dof : dof + 3].copy() for dof in env.food_dof_all]
    )
    initial_bowl_velocities = np.asarray(
        [env.data.qvel[dof : dof + 3].copy() for dof in env.moving_bowl_dof_all]
    )

    calls: list[dict[str, object]] = []
    for item_index in range(num_items):
        if item_index > 0:
            observation = env.prepare_static_sphere_moving_bowl_call(item_index)
        sphere_at_invocation = env.data.xpos[env.food_body].copy()
        bowl_at_invocation = env.data.xpos[env.destination_bowl_body].copy()
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
            raise RuntimeError("queue call produced no control steps")
        success = result.outcome == "strict_place_home_success"
        calls.append(
            {
                "call": item_index + 1,
                "sphere": item_index,
                "bowl": item_index,
                "invocation_world_time": round(invocation_world_time, 4),
                "sphere_position_at_invocation": sphere_at_invocation.tolist(),
                "bowl_position_at_invocation": bowl_at_invocation.tolist(),
                "outcome": result.outcome,
                "success": success,
                "policy_seconds": round(env.elapsed, 3),
                "network_control_steps": action_values // 7,
                "network_nonzero_action_steps": nonzero_action_steps,
                "network_mean_abs_action": action_abs_sum / max(action_values, 1),
                "network_max_abs_action": action_max_abs,
                "phases": phases,
                "final_info": result.info,
            }
        )
        if not success:
            break

    settle_world(env, 1.0)
    final_pairs = pair_snapshot(env)
    retained = sum(bool(record["retained_and_settled"]) for record in final_pairs)
    successes = sum(bool(call["success"]) for call in calls)
    belt_geom = env.model.geom("belt_visual").id
    belt_center = float(env.model.geom_pos[belt_geom, 0])
    belt_half_length = float(env.model.geom_size[belt_geom, 0])
    strict_success = bool(
        len(calls) == num_items and successes == num_items and retained == num_items
    )
    report = {
        "engine": "mujoco",
        "physics_closed_loop": True,
        "robot": "jaka_zu5_ag95",
        "task": "ten_static_spheres_to_ten_visible_moving_bowls",
        "controller": "frozen_checkpoint_residual_plus_geometry_baseline",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "network_called_every_control_step": True,
        "zero_residual_baseline": False,
        "translation_residual_limit": 0.03,
        "rotation_residual_limit": 0.015,
        "policy_calls_requested": num_items,
        "policy_calls_completed": len(calls),
        "successful_policy_calls": successes,
        "balls_retained_in_assigned_bowls": retained,
        "strict_sequence_success": strict_success,
        "bowl_speed_mps": bowl_speed,
        "bowl_spacing_m": spacing,
        "belt_upstream_x": belt_center - belt_half_length,
        "belt_downstream_x": belt_center + belt_half_length,
        "belt_length_m": 2.0 * belt_half_length,
        "all_spheres_present_at_start": bool(
            np.all(env.model.geom_contype[env.food_geom_all] != 0)
        ),
        "all_spheres_static_at_start": bool(
            np.all(np.linalg.norm(initial_sphere_velocities, axis=1) < 1.0e-9)
        ),
        "all_bowls_present_at_start": len(initial_bowl_positions) == num_items,
        "all_bowls_moving_at_start": bool(
            np.all(initial_bowl_velocities[:, 0] > 0.95 * bowl_speed)
        ),
        "spawned_between_calls": False,
        "world_reset_between_calls": False,
        "balls_removed_after_place": False,
        "bowls_removed_after_place": False,
        "grasp_assist": False,
        "actor_speed_input": False,
        "initial_sphere_positions": initial_sphere_positions.tolist(),
        "initial_bowl_positions": initial_bowl_positions.tolist(),
        "calls": calls,
        "final_pairs": final_pairs,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("JAKA_TEN_STATIC_TO_TEN_MOVING " + json.dumps(report), flush=True)
    env.close()


if __name__ == "__main__":
    main()
