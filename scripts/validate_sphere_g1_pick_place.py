"""Isaac geometry gate for G1 moving-sphere transport and release.

The frozen G0 actor remains speed/contact/force-free.  Simulator object state
is read only below for diagnostics and strict task acceptance; the controller
receives grasp geometry, TCP/finger state, and the calibrated static bowl pose.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from pickplace.app import launch_app


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--bowl-y", type=float, default=-0.10)
parser.add_argument(
    "--controller-profile",
    choices=("auto", "franka_g0_s2c", "jaka_j0_s3b"),
    default="auto",
)
parser.add_argument("--speed-min", type=float, default=0.08)
parser.add_argument("--speed-max", type=float, default=0.08)
parser.add_argument("--translation-residual-limit", type=float, default=None)
parser.add_argument("--rotation-residual-limit", type=float, default=None)
parser.add_argument("--randomize-position", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--zero-residual", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

app = launch_app(headless=True, enable_cameras=False)

import torch  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from pickplace.grasp_skills import (  # noqa: E402
    SphereGraspSkillConfig,
    SphereTransportPlaceConfig,
    StableSpherePickPlaceSkill,
    StableSphereResidualSkill,
    StableTargetConfig,
)
from pickplace.torchrl_env import make_env  # noqa: E402


def _teacher_utils():
    path = Path(__file__).resolve().parents[1] / "pipeline/0_state_teacher/utils.py"
    spec = importlib.util.spec_from_file_location("g1_pick_place_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_env = payload["config"]["env"]
    arm = str(checkpoint_env.get("arm", "franka"))
    profile = args.controller_profile
    if profile == "auto":
        profile = "jaka_j0_s3b" if arm == "zu5_ag95" else "franka_g0_s2c"
    if args.speed_min <= 0.0 or args.speed_max < args.speed_min:
        raise ValueError("speed bounds must satisfy 0 < speed_min <= speed_max")
    speed = 0.5 * (args.speed_min + args.speed_max)
    speed_noise = (args.speed_max - args.speed_min) / (args.speed_max + args.speed_min)
    jaka = profile == "jaka_j0_s3b"
    default_pallet = [-0.02, 0.02] if jaka else [-0.40, -0.36]
    fixed_pallet = [0.0, 0.0] if jaka else [-0.38, -0.38]
    spread = {
        "pallet_start_range": default_pallet if args.randomize_position else fixed_pallet,
        "bowl_offset_x": [-0.01, 0.01] if args.randomize_position else [0.0, 0.0],
        "bowl_offset_y": [-0.01, 0.01] if args.randomize_position else [0.0, 0.0],
    }
    overrides = {
        "task": "FoodRobot-MovingSphereStaticBowl-v0",
        "num_envs": args.num_envs,
        "seed": args.seed,
        "cameras": False,
        "reward_set": "moving_sphere_v1",
        "success_requires_home": False,
        "destination_bowl_pos": [0.45, args.bowl_y, 0.0],
        "destination_bowl_x_range": [0.44, 0.46]
        if args.randomize_position
        else [0.45, 0.45],
        "destination_bowl_y_range": [args.bowl_y - 0.01, args.bowl_y + 0.01]
        if args.randomize_position
        else [args.bowl_y, args.bowl_y],
        # The G1 controller is applied explicitly below so its phase can be
        # audited independently of TorchRL transform ordering.
        "stable_sphere_residual_skill": {"enabled": False},
        "stable_sphere_grasp_option": {"enabled": False},
        "stable_sphere_pick_place_skill": {"enabled": False},
        "belt": {
            "speed": speed,
            "speed_noise": speed_noise,
            "place_window": 0.40 / speed,
            "visual_upstream_extension": 0.0 if jaka else 0.25,
            # G0 needed 10 s after the interception path.  G1 additionally
            # transports to the side bowl, so reserve four seconds without
            # weakening any physical success or failure condition.
            "episode_completion_margin": 14.0,
            "pallet": {"travel_lower": -0.12 if jaka else -0.44},
            **spread,
        },
    }
    env_cfg = OmegaConf.to_container(
        OmegaConf.merge(payload["config"]["env"], overrides), resolve=True
    )
    env = make_env(env_cfg)
    actor = _teacher_utils().load_teacher_actor(args.checkpoint, env, env.device)
    actor.eval()
    residual_cfg = checkpoint_env.get("stable_sphere_residual_skill", {})
    residual_limit = (
        0.0
        if args.zero_residual
        else float(
            args.translation_residual_limit
            if args.translation_residual_limit is not None
            else residual_cfg.get("translation_residual_limit", 0.10)
        )
    )
    rotation_limit = (
        0.0
        if args.zero_residual
        else float(
            args.rotation_residual_limit
            if args.rotation_residual_limit is not None
            else residual_cfg.get("rotation_residual_limit", 0.05)
        )
    )
    close_action = -0.43 if jaka else -0.15
    grasp = StableSphereResidualSkill(
        SphereGraspSkillConfig(
            action_scale=0.02,
            intercept_x=0.45,
            intercept_lead_distance=0.16,
            intercept_align_lead_distance=0.06,
            hover_height=0.08,
            pre_descent_height=0.02,
            hover_translation_threshold=0.012,
            grasp_translation_threshold=0.007,
            rotation_threshold=0.15,
            alignment_steps=4,
            close_steps=16,
            hold_steps=10,
            lift_steps=60,
            lift_timeout_steps=300 if jaka else 1200,
            lift_height=0.05,
            lift_tolerance=0.003,
            lift_xy_gain=0.01,
            max_lift_xy_step=0.002,
            preserve_lift_relative_xy=True,
            target_xy_feedforward_gain=1.0 if jaka else 4.0,
            max_target_xy_feedforward_step=0.002 if jaka else 0.008,
            align_target_xy_feedforward=True,
            close_requires_raw_alignment=True,
            max_close_hold_xy_step=0.008,
            latch_lift_xy_feedforward=True,
            lift_feedforward_x_only=jaka,
            lift_feedforward_decay_steps=20 if jaka else None,
            lift_geometry_relatch_height=0.012,
            settled_coupling_reference=True,
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
        StableTargetConfig(),
        translation_residual_limit=residual_limit,
        rotation_residual_limit=rotation_limit,
    )
    skill = StableSpherePickPlaceSkill(
        grasp,
        SphereTransportPlaceConfig(
            action_scale=0.02,
            clear_height=0.20,
            bowl_hover_height=0.20,
            release_height=0.11,
            position_gain=4.0,
            max_position_step=0.016,
            max_translation_action_delta=0.04,
            clear_tolerance=0.012,
            hover_tolerance=0.018,
            release_tolerance=0.012,
            stable_steps=3,
            release_steps=28,
            close_action=close_action,
            open_action=1.0,
        ),
    )

    base = env.base_env._env.unwrapped
    scene = base.scene
    td = env.reset()
    n = args.num_envs
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(finished)
    outcome_names = ("cube_exited_zone", "cube_off_belt", "time_out")
    outcomes = {name: torch.zeros_like(finished) for name in outcome_names}
    max_grasp_phase = torch.zeros(n, dtype=torch.long, device=env.device)
    max_place_phase = torch.zeros(n, dtype=torch.long, device=env.device)
    initial_food_z = (scene["food"].data.root_pos_w.torch - scene.env_origins)[:, 2].clone()
    max_lift = torch.zeros(n, device=env.device)
    min_bowl_xy = torch.full((n,), float("inf"), device=env.device)
    min_tcp_bowl_distance = torch.full((n,), float("inf"), device=env.device)
    place_phase_entry_step = torch.full(
        (n, len(skill.PHASE_NAMES)), -1, dtype=torch.long, device=env.device
    )
    max_steps = int(base.max_episode_length) + 1

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for step in range(max_steps):
            live = ~finished
            food = scene["food"].data.root_pos_w.torch - scene.env_origins
            bowl = td["belt", "bowl_pos"]
            max_lift[live] = torch.maximum(
                max_lift[live], food[live, 2] - initial_food_z[live]
            )
            min_bowl_xy[live] = torch.minimum(
                min_bowl_xy[live],
                torch.linalg.vector_norm(food[live, :2] - bowl[live, :2], dim=-1),
            )

            td = actor(td)
            action, grasp_phase, place_phase = skill(
                td["action"],
                td["grasp_guidance", "candidate_pose_deltas"],
                td["proprio", "ee_pos"],
                td["proprio", "gripper_pos"],
                bowl,
            )
            for phase_id in range(len(skill.PHASE_NAMES)):
                first_entry = (place_phase == phase_id) & (place_phase_entry_step[:, phase_id] < 0)
                place_phase_entry_step[first_entry, phase_id] = step
            min_tcp_bowl_distance[live] = torch.minimum(
                min_tcp_bowl_distance[live],
                torch.linalg.vector_norm(td["proprio", "ee_pos"][live] - bowl[live], dim=-1),
            )
            action[finished] = 0.0
            action[finished, -1] = 1.0
            td.set("action", action)
            max_grasp_phase[live] = torch.maximum(max_grasp_phase[live], grasp_phase[live])
            max_place_phase[live] = torch.maximum(max_place_phase[live], place_phase[live])

            transition, td = env.step_and_maybe_reset(td)
            done = transition["next", "done"].reshape(-1) & live
            if done.any():
                success |= transition["next", "outcome", "success"].reshape(-1) & done
                for name in outcome_names:
                    outcomes[name] |= transition["next", "outcome", name].reshape(-1) & done
                finished |= done
                skill.reset(done.reshape(-1, 1))
            if finished.all():
                break

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_frames": int(payload["frames"]),
        "arm": arm,
        "controller_profile": profile,
        "controller": "g0_s2c_grasp_then_geometry_clear_transport_lower_release",
        "controller_inputs": [
            "6d_grasp_candidate_set",
            "tcp_pose",
            "finger_encoders",
            "static_bowl_pose",
        ],
        "forbidden_actor_and_controller_inputs": [
            "object_velocity",
            "belt_speed",
            "contact",
            "force",
            "tactile",
            "current",
            "torque",
        ],
        "episodes": n,
        "steps": step + 1,
        "randomize_position": args.randomize_position,
        "bowl_y_m": args.bowl_y,
        "speed_range_mps": [args.speed_min, args.speed_max],
        "zero_residual": args.zero_residual,
        "translation_residual_limit": residual_limit,
        "rotation_residual_limit": rotation_limit,
        "successes": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "outcomes": {name: int(mask.sum()) for name, mask in outcomes.items()},
        "phase_reached": {
            "grasp": {
                name: int((max_grasp_phase >= phase_id).sum())
                for phase_id, name in enumerate(grasp.baseline.PHASE_NAMES)
            },
            "place": {
                name: int((max_place_phase >= phase_id).sum())
                for phase_id, name in enumerate(skill.PHASE_NAMES)
            },
        },
        "max_true_lift_m": {
            "min": float(max_lift.min()),
            "mean": float(max_lift.mean()),
            "max": float(max_lift.max()),
        },
        "min_food_to_bowl_xy_m": {
            "min": float(min_bowl_xy.min()),
            "mean": float(min_bowl_xy.mean()),
            "max": float(min_bowl_xy.max()),
        },
        "min_tcp_to_bowl_3d_m": {
            "min": float(min_tcp_bowl_distance.min()),
            "mean": float(min_tcp_bowl_distance.mean()),
            "max": float(min_tcp_bowl_distance.max()),
        },
        "place_phase_entry_steps": {
            name: place_phase_entry_step[:, phase_id].cpu().tolist()
            for phase_id, name in enumerate(skill.PHASE_NAMES)
        },
        "final_tcp_xyz_m": td["proprio", "ee_pos"].cpu().tolist(),
        "final_food_xyz_m": (
            scene["food"].data.root_pos_w.torch - scene.env_origins
        ).cpu().tolist(),
        "final_bowl_xyz_m": td["belt", "bowl_pos"].cpu().tolist(),
        "diagnostic_only_privileged_values": ["food_position", "termination_flags"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SPHERE_G1_PICK_PLACE " + json.dumps(report), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
