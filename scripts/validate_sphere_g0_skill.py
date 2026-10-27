"""Strict Stage-G validation for the speed-free stable sphere grasp skill.

The controller reads only the deployable grasp pose stream, TCP pose, and
finger encoders.  Simulator food pose and finger contact are used exclusively
below for diagnostics and strict acceptance.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pickplace.app import launch_app


ARGS = dict(arg.split("=", 1) for arg in sys.argv[1:] if "=" in arg)
DEVICE = ARGS.get("device", "cuda:0")
APP = launch_app(headless=True, enable_cameras=False, device=DEVICE)

import torch  # noqa: E402

from pickplace.grasp_skills import (  # noqa: E402
    SphereGraspSkillConfig,
    StableSphereGraspSkill,
    StableTargetConfig,
)
from pickplace.torchrl_env import make_env  # noqa: E402


def _bool(name: str, default: bool) -> bool:
    return ARGS.get(name, str(default)).lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    return float(ARGS.get(name, str(default)))


def _optional_float(name: str, default: float | None) -> float | None:
    value = ARGS.get(name)
    if value is None:
        return default
    if value.lower() in ("none", "null"):
        return None
    return float(value)


def _range(prefix: str, default: list[float]) -> list[float]:
    """Read an independently configurable uniform range from CLI arguments."""
    return [
        _float(f"{prefix}_min", default[0]),
        _float(f"{prefix}_max", default[1]),
    ]


def main() -> None:
    num_envs = int(ARGS.get("num_envs", "20"))
    speed = float(ARGS.get("speed", "0.01"))
    arm = ARGS.get("arm", "franka")
    if arm not in ("franka", "zu5_ag95"):
        raise ValueError("arm must be franka or zu5_ag95")
    close_action = float(ARGS.get("close_action", "-0.43" if arm == "zu5_ag95" else "-0.15"))
    mode = ARGS.get("mode", "skill")
    if mode not in ("skill", "idle", "axis"):
        raise ValueError("mode must be skill, idle, or axis")
    randomize = _bool("randomize", False)
    canonical = _bool("canonical", False)
    seed = int(ARGS.get("seed", "31"))
    action_mode = ARGS.get("action_mode", "ee_delta_pose")
    if action_mode not in ("ee_delta_pose", "joint_pos"):
        raise ValueError("action_mode must be ee_delta_pose or joint_pos")
    output = Path(
        ARGS.get(
            "out",
            f"artifacts/moving_sphere_static_bowl/validation/{arm}_g0_skill_"
            f"{'canonical_' if canonical else 'multiyaw_'}v1i_{speed:.2f}_"
            f"{'random' if randomize else 'fixed'}_{num_envs}ep.json",
        )
    )
    spread = {
        "pallet_start_range": _range(
            "pallet_start", [-0.04, 0.04] if randomize else [0.0, 0.0]
        ),
        # These historical belt-config names map to food_offset_x/y in the
        # moving-object reset event; the static bowl is not randomized here.
        "bowl_offset_x": _range(
            "food_offset_x", [-0.01, 0.01] if randomize else [0.0, 0.0]
        ),
        "bowl_offset_y": _range(
            "food_offset_y", [-0.02, 0.02] if randomize else [0.0, 0.0]
        ),
    }
    speed_noise = _float("speed_noise", 0.5 if randomize else 0.0)
    robot_reset_position = _float(
        "robot_reset_position", 0.01 if randomize else 0.0
    )
    cfg = {
        "task": (
            "FoodRobot-MovingSphereCanonicalGapLift-v0"
            if canonical else "FoodRobot-MovingSphereGapLift-v0"
        ),
        "num_envs": num_envs,
        "arm": arm,
        "food": "moving_sphere",
        "action_mode": action_mode,
        "action_adapter": "native",
        "gripper_action_mode": "continuous",
        "ik_action_scale": 0.02,
        "cameras": False,
        "privileged_information": False,
        "grasp_observation": False,
        "grasp_point_observation": True,
        "speed_free_observations": True,
        "belt": {
            "speed": speed,
            "speed_noise": speed_noise,
            "place_window": 0.40 / speed,
            **spread,
        },
        "robot_reset": {
            "position_range": [-robot_reset_position, robot_reset_position],
            "velocity_range": [0.0, 0.0],
        },
        "target_lift": 0.03,
        "reward_set": "sphere_gap_lift_v1",
        "grasp_pose_close_retract_option": {"enabled": False},
        "device": DEVICE,
        "seed": seed,
    }
    env = make_env(cfg)
    td = env.reset()
    initial_tcp = td["proprio", "ee_pos"].clone()
    initial_tcp_quat = td["proprio", "ee_quat"].clone()
    initial_grasp_delta = td["grasp_guidance", "best_pose_delta"].clone()
    skill = StableSphereGraspSkill(
        SphereGraspSkillConfig(
            action_scale=0.02,
            intercept_x=0.45,
            intercept_lead_distance=_float("intercept_lead_distance", 0.08),
            intercept_align_lead_distance=_float("intercept_align_lead_distance", 0.04),
            hover_height=0.08,
            pre_descent_height=_optional_float("pre_descent_height", None),
            hover_translation_threshold=0.012,
            grasp_translation_threshold=0.007,
            rotation_threshold=0.15,
            alignment_steps=4,
            close_steps=16,
            hold_steps=10,
            lift_steps=int(ARGS.get("lift_steps", "30")),
            lift_timeout_steps=int(ARGS.get("lift_timeout_steps", "150")),
            lift_height=_float("lift_height", 0.04),
            lift_tolerance=0.003,
            lift_target_slack=_optional_float("lift_target_slack", None),
            lift_detach_height=_optional_float("lift_detach_height", None),
            lift_xy_gain=_optional_float("lift_xy_gain", None),
            max_lift_xy_step=_optional_float("max_lift_xy_step", None),
            preserve_lift_relative_xy=_bool("preserve_lift_relative_xy", False),
            target_xy_feedforward_gain=_float("target_xy_feedforward_gain", 0.0),
            max_target_xy_feedforward_step=_optional_float(
                "max_target_xy_feedforward_step", None
            ),
            align_target_xy_feedforward=_bool("align_target_xy_feedforward", False),
            close_requires_raw_alignment=_bool("close_requires_raw_alignment", False),
            max_close_hold_xy_step=_optional_float("max_close_hold_xy_step", None),
            latch_lift_xy_feedforward=_bool("latch_lift_xy_feedforward", False),
            lift_feedforward_x_only=_bool("lift_feedforward_x_only", False),
            lift_feedforward_decay_steps=(
                None
                if ARGS.get("lift_feedforward_decay_steps", "null").lower() in ("none", "null")
                else int(ARGS["lift_feedforward_decay_steps"])
            ),
            lift_geometry_relatch_height=_optional_float(
                "lift_geometry_relatch_height", None
            ),
            lift_relative_z_gain=_optional_float("lift_relative_z_gain", None),
            settled_coupling_reference=_bool("settled_coupling_reference", False),
            position_gain=1.5,
            align_position_gain=4.0,
            rotation_gain=0.5,
            max_hover_position_step=0.02,
            max_hover_rotation_step=0.02,
            align_fine_distance=0.02,
            max_align_position_step=0.02,
            max_align_fine_position_step=0.008,
            max_position_step=0.004,
            max_rotation_step=0.01,
            close_action=close_action,
        ),
        StableTargetConfig(
            translation_alpha=0.45,
            rotation_alpha=0.35,
            max_translation_jump=0.02,
            max_rotation_jump=0.35,
            stable_translation=0.004,
            stable_rotation=0.08,
            stable_steps=4,
            rotation_weight=0.05,
            switch_margin=0.01,
        ),
    )

    scene = env.base_env._env.unwrapped.scene
    robot = scene["robot"]
    ee_body_id = robot.body_names.index(env.base_env._env.unwrapped.cfg.arm.ee_body_name)
    initial_joint_pos = robot.data.joint_pos.torch.clone()
    initial_ee_body_pos = (robot.data.body_pos_w.torch[:, ee_body_id] - scene.env_origins).clone()
    initial_ee_body_quat = robot.data.body_quat_w.torch[:, ee_body_id].clone()
    initial_body_pos = (robot.data.body_pos_w.torch - scene.env_origins[:, None, :]).clone()
    min_tcp = initial_tcp.clone()
    max_tcp = initial_tcp.clone()
    sampled_speed = env.base_env._env.unwrapped.event_manager.get_term_cfg("reset_belt").func.speed.clone()
    finished = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(finished)
    outcomes = {
        name: torch.zeros_like(finished)
        for name in ("cube_exited_zone", "cube_off_belt", "time_out")
    }
    phase_reached = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    phase_steps = torch.zeros(num_envs, len(skill.PHASE_NAMES), dtype=torch.long, device=env.device)
    close_reference = torch.zeros(num_envs, 3, device=env.device)
    close_captured = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    max_coupling_drift = torch.zeros(num_envs, device=env.device)
    contact_reference = torch.zeros(num_envs, 3, device=env.device)
    contact_reference_valid = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    max_contact_coupling_drift = torch.zeros(num_envs, device=env.device)
    max_lift = torch.zeros(num_envs, device=env.device)
    max_surface_clearance_lift = torch.zeros(num_envs, device=env.device)
    lift_start_tcp_z = torch.zeros(num_envs, device=env.device)
    max_tcp_lift = torch.zeros(num_envs, device=env.device)
    max_left_force = torch.zeros(num_envs, device=env.device)
    max_right_force = torch.zeros(num_envs, device=env.device)
    simultaneous_contact = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    min_finger = torch.full((num_envs,), float("inf"), device=env.device)
    done_step = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
    initial_food_z = scene["food"].data.root_pos_w.torch[:, 2].clone()
    initial_food = (scene["food"].data.root_pos_w.torch - scene.env_origins).clone()
    min_food = initial_food.clone()
    max_food = initial_food.clone()

    max_steps = min(
        int(env.base_env._env.unwrapped.max_episode_length),
        int(ARGS.get("max_steps", str(env.base_env._env.unwrapped.max_episode_length))),
    )
    trace_every = int(ARGS.get("trace_every", "0"))
    if trace_every > 0:
        jac = robot.data.body_link_jacobian_w.torch
        print(
            "JAKA_JACOBIAN",
            json.dumps(
                {
                    "shape": list(jac.shape),
                    "row_norms": torch.linalg.vector_norm(jac[0], dim=(-2, -1)).detach().cpu().tolist(),
                    "body_names": list(robot.body_names),
                }
            ),
            flush=True,
        )
    for step in range(max_steps):
        base = torch.zeros(num_envs, 7, device=env.device)
        if mode == "idle":
            action = base
            action[:, -1] = 1.0
            phase = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        elif mode == "axis":
            action = base
            action[:, int(ARGS.get("axis", "2"))] = float(ARGS.get("axis_action", "0.5"))
            action[:, -1] = 1.0
            phase = torch.zeros(num_envs, dtype=torch.long, device=env.device)
        else:
            action, phase, _ = skill(
                base,
                td["grasp_guidance", "candidate_pose_deltas"],
                td["proprio", "ee_pos"],
                td["proprio", "gripper_pos"],
            )
        action[finished] = 0.0
        action[finished, -1] = 1.0
        phase_reached = torch.maximum(phase_reached, phase)
        for phase_id in range(len(skill.PHASE_NAMES)):
            phase_steps[:, phase_id] += ((phase == phase_id) & ~finished).long()

        food_local = scene["food"].data.root_pos_w.torch - scene.env_origins
        min_tcp = torch.minimum(min_tcp, td["proprio", "ee_pos"])
        max_tcp = torch.maximum(max_tcp, td["proprio", "ee_pos"])
        min_food = torch.minimum(min_food, food_local)
        max_food = torch.maximum(max_food, food_local)
        max_lift = torch.maximum(max_lift, food_local[:, 2] - initial_food_z)
        surface_clearance_lift = (
            food_local[:, 2]
            - float(env.base_env._env.unwrapped.cfg.belt.plate_top_z)
            - float(env.base_env._env.unwrapped.cfg.food.item_radius)
        ).clamp_min(0.0)
        max_surface_clearance_lift = torch.maximum(
            max_surface_clearance_lift, surface_clearance_lift
        )
        relative = food_local - td["proprio", "ee_pos"]
        entering_close = (phase >= skill.CLOSE) & ~close_captured
        close_reference[entering_close] = relative[entering_close]
        close_captured |= entering_close
        entering_lift = (phase == skill.LIFT) & (phase_steps[:, skill.LIFT] == 1)
        lift_start_tcp_z[entering_lift] = td["proprio", "ee_pos"][entering_lift, 2]
        tcp_lift = td["proprio", "ee_pos"][:, 2] - lift_start_tcp_z
        max_tcp_lift = torch.where(
            (phase >= skill.LIFT) & close_captured & ~finished,
            torch.maximum(max_tcp_lift, tcp_lift),
            max_tcp_lift,
        )
        coupled_phase = (phase >= skill.CLOSE) & ~finished & close_captured
        drift = torch.linalg.vector_norm(relative - close_reference, dim=-1)
        max_coupling_drift = torch.where(
            coupled_phase, torch.maximum(max_coupling_drift, drift), max_coupling_drift
        )

        td.set("action", action)
        transition, td = env.step_and_maybe_reset(td)
        if trace_every > 0 and (step % trace_every == 0 or not torch.isfinite(robot.data.joint_pos.torch).all()):
            print(
                "JAKA_TRACE",
                json.dumps(
                    {
                        "step": step + 1,
                        "joint_pos": robot.data.joint_pos.torch[0].detach().cpu().tolist(),
                        "joint_target": robot.data.joint_pos_target.torch[0].detach().cpu().tolist(),
                        "food": (
                            scene["food"].data.root_pos_w.torch[0] - scene.env_origins[0]
                        ).detach().cpu().tolist(),
                        "tcp": td["proprio", "ee_pos"][0].detach().cpu().tolist(),
                        "gripper_obs": td["proprio", "gripper_pos"][0].detach().cpu().tolist(),
                    }
                ),
                flush=True,
            )
        forces = []
        for name, maximum in (
            ("left_finger_contact", max_left_force),
            ("right_finger_contact", max_right_force),
        ):
            force = scene[name].data.force_matrix_w.norm(dim=-1).amax(dim=(-1, -2))
            maximum.copy_(torch.maximum(maximum, force))
            forces.append(force)
        simultaneous_contact |= (forces[0] > 0.05) & (forces[1] > 0.05)
        held_now = (forces[0] > 0.05) & (forces[1] > 0.05)
        food_after = scene["food"].data.root_pos_w.torch - scene.env_origins
        relative_after = food_after - td["proprio", "ee_pos"]
        first_contact = held_now & ~contact_reference_valid
        contact_reference[first_contact] = relative_after[first_contact]
        contact_reference_valid |= first_contact
        contact_drift = torch.linalg.vector_norm(relative_after - contact_reference, dim=-1)
        max_contact_coupling_drift = torch.where(
            held_now,
            torch.maximum(max_contact_coupling_drift, contact_drift),
            max_contact_coupling_drift,
        )
        min_finger = torch.minimum(min_finger, td["proprio", "gripper_pos"].mean(-1))

        done = transition["next", "done"].reshape(-1) & ~finished
        if done.any():
            done_step[done] = step + 1
            success |= transition["next", "outcome", "success"].reshape(-1) & done
            for name in outcomes:
                outcomes[name] |= transition["next", "outcome", name].reshape(-1) & done
            finished |= done
        if finished.all():
            break

    report = {
        "controller": "fixed_intercept_close_hold_lift_g0_v1i",
        "mode": mode,
        "arm": arm,
        "close_action": close_action,
        "controller_inputs": ["6d_grasp_candidate_set", "tcp_position", "finger_encoders"],
        "forbidden_controller_inputs": [
            "object_velocity", "belt_speed", "contact", "force", "tactile", "current", "torque"
        ],
        "acceptance": (
            "stable dual-finger hold, sphere-TCP drift <= 0.008 m, "
            "and true sphere lift >= 0.03 m"
        ),
        "episodes": num_envs,
        "nominal_speed_mps": speed,
        "randomize": randomize,
        "environment_randomization": {
            "speed_noise_fraction": speed_noise,
            "speed_range_mps": [speed * (1.0 - speed_noise), speed * (1.0 + speed_noise)],
            "pallet_start_range_m": spread["pallet_start_range"],
            "food_offset_x_range_m": spread["bowl_offset_x"],
            "food_offset_y_range_m": spread["bowl_offset_y"],
            "robot_reset_position_range_rad": [
                -robot_reset_position,
                robot_reset_position,
            ],
        },
        "canonical_sphere_yaw": canonical,
        "seed": seed,
        "skill_overrides": {
            key: value
            for key, value in ARGS.items()
            if key
            in {
                "intercept_lead_distance", "intercept_align_lead_distance",
                "pre_descent_height", "lift_steps", "lift_timeout_steps",
                "lift_height", "lift_target_slack", "lift_detach_height",
                "lift_xy_gain", "max_lift_xy_step", "preserve_lift_relative_xy",
                "target_xy_feedforward_gain", "max_target_xy_feedforward_step",
                "align_target_xy_feedforward", "close_requires_raw_alignment",
                "max_close_hold_xy_step", "latch_lift_xy_feedforward",
                "lift_feedforward_x_only", "lift_feedforward_decay_steps",
                "lift_geometry_relatch_height", "lift_relative_z_gain",
                "settled_coupling_reference",
            }
        },
        "successes": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "steps": step + 1,
        "initial_tcp": initial_tcp.cpu().tolist(),
        "final_tcp": td["proprio", "ee_pos"].cpu().tolist(),
        "min_tcp": min_tcp.cpu().tolist(),
        "max_tcp": max_tcp.cpu().tolist(),
        "initial_tcp_quat": initial_tcp_quat.cpu().tolist(),
        "initial_joint_pos": initial_joint_pos.cpu().tolist(),
        "final_joint_pos": robot.data.joint_pos.torch.cpu().tolist(),
        "final_joint_target": robot.data.joint_pos_target.torch.cpu().tolist(),
        "runtime_joint_names": list(robot.joint_names),
        "initial_ee_body_pos": initial_ee_body_pos.cpu().tolist(),
        "initial_ee_body_quat": initial_ee_body_quat.cpu().tolist(),
        "initial_body_names": list(robot.body_names),
        "initial_body_pos": initial_body_pos.cpu().tolist(),
        "initial_grasp_delta": initial_grasp_delta.cpu().tolist(),
        "initial_food": initial_food.cpu().tolist(),
        "min_food": min_food.cpu().tolist(),
        "max_food": max_food.cpu().tolist(),
        **{name: int(mask.sum()) for name, mask in outcomes.items()},
        "phase_reached": {
            name: int((phase_reached >= phase_id).sum())
            for phase_id, name in enumerate(skill.PHASE_NAMES)
        },
        "simultaneous_dual_contact": int(simultaneous_contact.sum()),
        "strict_surface_clearance_lift_30mm": int(success.sum()),
        "max_surface_clearance_lift_mean_m": float(max_surface_clearance_lift.mean()),
        "observed_pre_autoreset_delta_lift_1mm": int((max_lift >= 0.001).sum()),
        "observed_pre_autoreset_delta_lift_10mm": int((max_lift >= 0.010).sum()),
        "observed_pre_autoreset_max_delta_lift_mean_m": float(max_lift.mean()),
        "max_tcp_lift_mean_m": float(max_tcp_lift.mean()),
        "max_close_phase_coupling_drift_mean_m": float(max_coupling_drift.mean()),
        "max_contact_coupling_drift_mean_m": float(max_contact_coupling_drift.mean()),
        "min_finger_mean_m": float(min_finger.mean()),
        "per_env": [
            {
                "speed_mps": float(sampled_speed[i]),
                "initial_food_position_m": initial_food[i].cpu().tolist(),
                "success": bool(success[i]),
                "outcome": next((name for name, mask in outcomes.items() if bool(mask[i])), "success" if success[i] else "unfinished"),
                "max_phase": skill.PHASE_NAMES[int(phase_reached[i])],
                "phase_steps": {
                    name: int(phase_steps[i, phase_id])
                    for phase_id, name in enumerate(skill.PHASE_NAMES)
                },
                "simultaneous_dual_contact": bool(simultaneous_contact[i]),
                "max_lift_m": float(max_lift[i]),
                "max_surface_clearance_lift_m": float(max_surface_clearance_lift[i]),
                "max_tcp_lift_m": float(max_tcp_lift[i]),
                "max_close_phase_coupling_drift_m": float(max_coupling_drift[i]),
                "max_contact_coupling_drift_m": float(max_contact_coupling_drift[i]),
                "max_left_force_n": float(max_left_force[i]),
                "max_right_force_n": float(max_right_force[i]),
                "min_finger_m": float(min_finger[i]),
                "done_step": int(done_step[i]),
            }
            for i in range(num_envs)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SPHERE_G0_SKILL " + json.dumps(report), flush=True)
    env.close()
    os._exit(0 if report["successes"] == num_envs else 2)


if __name__ == "__main__":
    main()
