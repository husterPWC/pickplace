"""Closed-loop validation of the analytic 6D grasp and close-retract option."""

from __future__ import annotations

import json
import os

from pickplace.app import launch_app

app = launch_app(headless=True, enable_cameras=False, device="cuda:0")

import torch  # noqa: E402

from pickplace.torchrl_env import make_env  # noqa: E402


def main():
    num_envs = int(os.environ.get("SPHERE_GAP_VALIDATION_ENVS", "20"))
    cfg = {
        "task": "FoodRobot-MovingSphereGapLift-v0",
        "num_envs": num_envs,
        "arm": "franka",
        "food": "moving_sphere",
        "action_mode": "ee_delta_pose",
        "action_adapter": "native",
        "gripper_action_mode": "continuous",
        "ik_action_scale": 0.02,
        "cameras": False,
        "privileged_information": False,
        "grasp_observation": False,
        "grasp_point_observation": True,
        "speed_free_observations": True,
        "belt": {
            "speed": 0.02,
            "speed_noise": 0.5,
            "place_window": 20.0,
            "pallet_start_range": [-0.04, 0.04],
            "bowl_offset_x": [-0.01, 0.01],
            "bowl_offset_y": [-0.02, 0.02],
        },
        "target_lift": 0.03,
        "reward_set": "sphere_gap_lift_v1",
        "grasp_pose_close_retract_option": {"enabled": True},
        "device": "cuda:0",
        "seed": 17,
    }
    env = make_env(cfg)
    td = env.reset()
    phase = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    finished = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(finished)
    outcomes = {name: torch.zeros_like(finished) for name in ("cube_exited_zone", "cube_off_belt", "time_out")}
    min_translation = torch.full((num_envs,), float("inf"), device=env.device)
    min_hover_error = torch.full((num_envs,), float("inf"), device=env.device)
    best_hover_joint_pos = torch.zeros(num_envs, 7, device=env.device)
    best_hover_tcp_pos = torch.zeros(num_envs, 3, device=env.device)
    best_hover_food_pos = torch.zeros(num_envs, 3, device=env.device)
    min_rotation = torch.full((num_envs,), float("inf"), device=env.device)
    max_lift = torch.full((num_envs,), -float("inf"), device=env.device)
    max_left_force = torch.zeros(num_envs, device=env.device)
    max_right_force = torch.zeros(num_envs, device=env.device)
    min_finger = torch.full((num_envs,), float("inf"), device=env.device)
    simultaneous_contact = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    max_tcp_lift = torch.zeros(num_envs, device=env.device)
    trigger_tcp_z = torch.zeros(num_envs, device=env.device)
    scene = env.base_env._env.unwrapped.scene
    sampled_speed = env.base_env._env.unwrapped.event_manager.get_term_cfg("reset_belt").func.speed.clone()
    option_controller = next(
        transform.controller
        for transform in env.transform.transforms
        if hasattr(transform, "controller") and transform.controller.__class__.__name__ == "GraspPoseCloseRetractOption"
    )
    max_option_phase = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    max_tcp_option_lift = torch.zeros(num_envs, device=env.device)
    done_step = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
    max_steps = int(env.base_env._env.unwrapped.max_episode_length)
    for step in range(max_steps):
        delta = td["grasp_guidance", "best_pose_delta"]
        hover_delta = delta[:, :3].clone()
        hover_delta[:, 2] += 0.08
        hover_error = hover_delta.norm(dim=-1)
        better_hover = hover_error < min_hover_error
        if better_hover.any():
            robot_q = scene["robot"].data.joint_pos.torch[:, :7]
            min_hover_error[better_hover] = hover_error[better_hover]
            best_hover_joint_pos[better_hover] = robot_q[better_hover]
            best_hover_tcp_pos[better_hover] = td["proprio", "ee_pos"][better_hover]
            best_hover_food_pos[better_hover] = (
                scene["food"].data.root_pos_w.torch[better_hover] - scene.env_origins[better_hover]
            )
        min_translation = torch.minimum(min_translation, delta[:, :3].norm(dim=-1))
        min_rotation = torch.minimum(min_rotation, delta[:, 3:].norm(dim=-1))
        food_z = scene["food"].data.root_pos_w.torch[:, 2] - scene.env_origins[:, 2]
        max_lift = torch.maximum(max_lift, food_z - 0.05)
        for name, target in (("left_finger_contact", max_left_force), ("right_finger_contact", max_right_force)):
            force = scene[name].data.force_matrix_w.norm(dim=-1).amax(dim=(-1, -2))
            target.copy_(torch.maximum(target, force))
        min_finger = torch.minimum(min_finger, td["proprio", "gripper_pos"].mean(-1))
        both_now = (
            scene["left_finger_contact"].data.force_matrix_w.norm(dim=-1).amax(dim=(-1, -2)) > 0.05
        ) & (
            scene["right_finger_contact"].data.force_matrix_w.norm(dim=-1).amax(dim=(-1, -2)) > 0.05
        )
        simultaneous_contact |= both_now
        entering = (phase == 0) & (hover_error < 0.012) & (delta[:, 3:].norm(dim=-1) < 0.15)
        trigger_tcp_z[entering] = td["proprio", "ee_pos"][entering, 2]
        current_tcp_lift = td["proprio", "ee_pos"][:, 2] - trigger_tcp_z
        max_tcp_lift = torch.where(phase > 0, torch.maximum(max_tcp_lift, current_tcp_lift), max_tcp_lift)
        phase[(phase == 0) & (hover_error < 0.012) & (delta[:, 3:].norm(dim=-1) < 0.15)] = 1
        translation = torch.where(phase[:, None] == 0, hover_delta, delta[:, :3])
        action = torch.zeros(num_envs, 7, device=env.device)
        action[:, :3] = (0.5 * translation / 0.02).clamp(-1.0, 1.0)
        action[:, 3:6] = (0.5 * delta[:, 3:] / 0.02).clamp(-1.0, 1.0)
        action[:, -1] = 1.0
        action[finished] = 0.0
        td.set("action", action)
        transition, td = env.step_and_maybe_reset(td)
        if option_controller.phase is not None:
            max_option_phase = torch.maximum(max_option_phase, option_controller.phase)
            active_option = option_controller.phase > (option_controller.approach_steps + option_controller.close_steps)
            option_lift = td["proprio", "ee_pos"][:, 2] - option_controller.anchor_tcp_z
            max_tcp_option_lift = torch.where(
                active_option & ~finished, torch.maximum(max_tcp_option_lift, option_lift), max_tcp_option_lift
            )
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
        "episodes": num_envs,
        "successes": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "steps": step + 1,
        **{name: int(mask.sum()) for name, mask in outcomes.items()},
        "descend_phase": int((phase > 0).sum()),
        "min_translation_mean": float(min_translation.mean()),
        "min_translation_max": float(min_translation.max()),
        "min_rotation_mean": float(min_rotation.mean()),
        "min_rotation_max": float(min_rotation.max()),
        "max_lift_mean": float(max_lift.mean()),
        "max_lift_max": float(max_lift.max()),
        "both_fingers_ever_contact": int(((max_left_force > 0.05) & (max_right_force > 0.05)).sum()),
        "simultaneous_dual_contact": int(simultaneous_contact.sum()),
        "max_tcp_lift_mean": float(max_tcp_lift.mean()),
        "max_option_phase": int(max_option_phase.max()),
        "max_tcp_option_lift_mean": float(max_tcp_option_lift.mean()),
        "min_finger_mean": float(min_finger.mean()),
        # Calibration aid for minimal-overfit curricula: this is a physically
        # reached joint pose above the moving sphere, not an IK guess or a
        # teleported object.  Keeping it in the validation report makes the
        # provenance of any near-grasp reset explicit and reproducible.
        "min_hover_error": min_hover_error.detach().cpu().tolist(),
        "best_hover_joint_pos": best_hover_joint_pos.detach().cpu().tolist(),
        "best_hover_tcp_pos": best_hover_tcp_pos.detach().cpu().tolist(),
        "best_hover_food_pos": best_hover_food_pos.detach().cpu().tolist(),
        "per_env": [
            {
                "speed": float(sampled_speed[i]),
                "success": bool(success[i]),
                "min_translation": float(min_translation[i]),
                "max_lift": float(max_lift[i]),
                "max_left_force": float(max_left_force[i]),
                "max_right_force": float(max_right_force[i]),
                "max_option_phase": int(max_option_phase[i]),
                "max_tcp_option_lift": float(max_tcp_option_lift[i]),
                "done_step": int(done_step[i]),
            }
            for i in range(num_envs)
        ],
    }
    print("SPHERE_GAP_VALIDATION " + json.dumps(report), flush=True)
    env.close()
    os._exit(0 if report["successes"] == num_envs else 2)


if __name__ == "__main__":
    main()
