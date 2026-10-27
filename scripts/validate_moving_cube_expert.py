"""Run the closed-loop moving-cube expert and save an acceptance report.

Examples:
    .venv/bin/python scripts/validate_moving_cube_expert.py speed=0.02 num_envs=8 randomize=false
    .venv/bin/python scripts/validate_moving_cube_expert.py speed=0.10 num_envs=20 randomize=true
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pickplace.app import launch_app


ARGS = dict(arg.split("=", 1) for arg in sys.argv[1:] if "=" in arg)
DEVICE = ARGS.get("device", "cuda:0")
DEFAULT_TASK = (
    "FoodRobot-MovingSphereStaticBowl-v0"
    if ARGS.get("food", "cube") == "moving_sphere"
    else "FoodRobot-MovingCubeStaticBowl-v0"
)
TASK = ARGS.get("task", DEFAULT_TASK)
APP = launch_app(headless=True, enable_cameras=False, device=DEVICE)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import pickplace.envs  # noqa: E402,F401
from pickplace.config import build_cell_env_cfg  # noqa: E402
from pickplace.experts import MovingCubeExpert, PHASE_NAMES  # noqa: E402


def _bool(name: str, default: bool) -> bool:
    return ARGS.get(name, str(default)).lower() in ("1", "true", "yes", "on")


def _masked_stats(values: torch.Tensor, mask: torch.Tensor) -> dict[str, object]:
    """Return compact per-axis statistics without hiding the raw samples."""
    selected = values[mask]
    if selected.numel() == 0:
        return {"count": 0, "mean": None, "min": None, "max": None, "samples": []}
    return {
        "count": int(selected.shape[0]),
        "mean": selected.mean(dim=0).tolist(),
        "min": selected.amin(dim=0).tolist(),
        "max": selected.amax(dim=0).tolist(),
        "samples": selected.tolist(),
    }


def main() -> None:
    speed = float(ARGS.get("speed", "0.02"))
    num_envs = int(ARGS.get("num_envs", "8"))
    randomize = _bool("randomize", False)
    seed = int(ARGS.get("seed", "0"))
    food = ARGS.get("food", "cube")
    food_prefix = "" if food == "cube" else f"{food}_"
    output = Path(
        ARGS.get(
            "out",
            f"artifacts/moving_cube_static_bowl/validation/{food_prefix}expert_{speed:.2f}_"
            f"{'random' if randomize else 'fixed'}.json",
        )
    )
    spread = {
        "pallet_start_range": [-0.04, 0.04] if randomize else [0.0, 0.0],
        "bowl_offset_x": [-0.01, 0.01] if randomize else [0.0, 0.0],
        "bowl_offset_y": [-0.02, 0.02] if randomize else [0.0, 0.0],
    }
    cfg = build_cell_env_cfg(
        {
            "task": TASK,
            "num_envs": num_envs,
            "arm": "franka",
            "food": food,
            "action_mode": "ee_delta_pose",
            "cameras": False,
            "privileged_information": True,
            "belt": {
                "speed": speed,
                "speed_noise": 0.05 if randomize else 0.0,
                # Preserve the same 0.40 m reach zone at every validation speed.
                "place_window": 0.40 / speed,
                **spread,
            },
            "robot_reset": {
                "position_range": [-0.01, 0.01] if randomize else [0.0, 0.0],
                "velocity_range": [0.0, 0.0],
            },
            "destination_bowl_pos": [0.45, -0.10, 0.0],
            "destination_bowl_x_range": [0.45, 0.45],
            "destination_bowl_y_range": [-0.10, -0.10],
            "reward_set": "moving_cube_v1",
            "success_requires_home": True,
            "seed": seed,
            "device": DEVICE,
        }
    )
    env = gym.make(TASK, cfg=cfg)
    unwrapped = env.unwrapped
    obs, _ = env.reset(seed=seed)
    expert = MovingCubeExpert(num_envs, unwrapped.device, unwrapped.step_dt, unwrapped.action_manager.total_action_dim)

    finished = torch.zeros(num_envs, dtype=torch.bool, device=unwrapped.device)
    outcome = ["unfinished"] * num_envs
    finish_phase = [-1] * num_envs
    max_cube_z = obs["privileged"]["food_pos"][:, 2].clone()
    ever_grasped = obs["privileged"]["is_grasped"].reshape(num_envs).bool().clone()
    min_bowl_xy = torch.full((num_envs,), float("inf"), device=unwrapped.device)
    # Capture the actual geometry used by the already-proven controller.  In
    # particular, the TCP frame origin is not assumed to be the finger-pad
    # centre.  These values are diagnostic/calibration outputs only; they are
    # not fed back into the deployable policy.
    lift_start_captured = torch.zeros(num_envs, dtype=torch.bool, device=unwrapped.device)
    lift_start_object_to_tcp = torch.full((num_envs, 3), float("nan"), device=unwrapped.device)
    lift_start_tcp = torch.full((num_envs, 3), float("nan"), device=unwrapped.device)
    lift_start_tcp_quat = torch.full((num_envs, 4), float("nan"), device=unwrapped.device)
    lift_start_fingers = torch.full((num_envs, 2), float("nan"), device=unwrapped.device)
    grasp_captured = torch.zeros(num_envs, dtype=torch.bool, device=unwrapped.device)
    first_grasp_object_to_tcp = torch.full((num_envs, 3), float("nan"), device=unwrapped.device)
    first_grasp_fingers = torch.full((num_envs, 2), float("nan"), device=unwrapped.device)
    max_steps = int(cfg.episode_length_s / unwrapped.step_dt) + 10

    for step in range(max_steps):
        previous_phase = expert.phase.clone()
        action = expert.act(obs)
        entered_lift = (previous_phase == 3) & (expert.phase == 4) & ~lift_start_captured
        if bool(entered_lift.any()):
            lift_start_object_to_tcp[entered_lift] = (
                obs["privileged"]["food_pos"][entered_lift] - obs["proprio"]["ee_pos"][entered_lift]
            )
            lift_start_tcp[entered_lift] = obs["proprio"]["ee_pos"][entered_lift]
            lift_start_tcp_quat[entered_lift] = obs["proprio"]["ee_quat"][entered_lift]
            lift_start_fingers[entered_lift] = obs["proprio"]["gripper_pos"][entered_lift]
            lift_start_captured |= entered_lift
        obs, _, terminated, truncated, _ = env.step(action)
        max_cube_z = torch.maximum(max_cube_z, obs["privileged"]["food_pos"][:, 2])
        grasped_now = obs["privileged"]["is_grasped"].reshape(num_envs).bool()
        first_grasp = grasped_now & ~grasp_captured
        if bool(first_grasp.any()):
            first_grasp_object_to_tcp[first_grasp] = (
                obs["privileged"]["food_pos"][first_grasp] - obs["proprio"]["ee_pos"][first_grasp]
            )
            first_grasp_fingers[first_grasp] = obs["proprio"]["gripper_pos"][first_grasp]
            grasp_captured |= first_grasp
        ever_grasped |= grasped_now
        bowl_xy = torch.linalg.vector_norm(
            obs["privileged"]["food_pos"][:, :2] - obs["belt"]["bowl_pos"][:, :2], dim=-1
        )
        min_bowl_xy = torch.minimum(min_bowl_xy, bowl_xy)

        done = (terminated | truncated).reshape(num_envs)
        newly_done = done & ~finished
        if bool(newly_done.any()):
            active_terms = {
                name: unwrapped.termination_manager.get_term(name).reshape(num_envs).clone()
                for name in ("success", "cube_exited_zone", "cube_off_belt", "time_out")
            }
            for index in newly_done.nonzero(as_tuple=False).flatten().tolist():
                fired = [name for name, flags in active_terms.items() if bool(flags[index])]
                outcome[index] = "+".join(fired) if fired else "unknown_done"
                finish_phase[index] = int(expert.phase[index])
            finished |= newly_done
            expert.mark_finished(newly_done)
        if bool(finished.all()):
            break

    report = {
        "task": TASK,
        "food": food,
        "controller": "closed_loop_current_state_expert_v1",
        "speed_mps": speed,
        "randomize": randomize,
        "seed": seed,
        "num_envs": num_envs,
        "steps": step + 1,
        "sim_seconds": (step + 1) * unwrapped.step_dt,
        "successes": sum(value == "success" for value in outcome),
        "success_rate": sum(value == "success" for value in outcome) / num_envs,
        "outcomes": outcome,
        "phase_at_finish": [PHASE_NAMES[p] if p >= 0 else "unfinished" for p in finish_phase],
        "last_phase": [PHASE_NAMES[p] for p in expert.phase.tolist()],
        "ever_grasped": ever_grasped.tolist(),
        "max_cube_z_m": max_cube_z.tolist(),
        "min_cube_to_bowl_xy_m": min_bowl_xy.tolist(),
        "pinch_geometry": {
            "convention": "object_position_minus_tcp_position_in_cell_frame",
            "lift_start_object_to_tcp_m": _masked_stats(
                lift_start_object_to_tcp, lift_start_captured
            ),
            "lift_start_tcp_position_m": _masked_stats(lift_start_tcp, lift_start_captured),
            "lift_start_tcp_quaternion_xyzw": _masked_stats(
                lift_start_tcp_quat, lift_start_captured
            ),
            "lift_start_finger_joint_position_m": _masked_stats(
                lift_start_fingers, lift_start_captured
            ),
            "first_is_grasped_object_to_tcp_m": _masked_stats(
                first_grasp_object_to_tcp, grasp_captured
            ),
            "first_is_grasped_finger_joint_position_m": _masked_stats(
                first_grasp_fingers, grasp_captured
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("MOVING_CUBE_EXPERT " + json.dumps(report), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
