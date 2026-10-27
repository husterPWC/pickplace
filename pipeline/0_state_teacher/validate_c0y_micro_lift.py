"""Validate the speed-free geometric micro-lift before training C0y.

Two first-episode tests share the same strict 5 mm / 5 step success term:

``c0u_gate``
    The frozen C0u deterministic actor intercepts and closes; the position-only
    micro-lift option takes over after five valid close observations.

``position_oracle``
    A current-position-only servo centers the gripper and commands closure;
    the exact same micro-lift option performs the retract.  It consumes no
    object velocity, force, contact, tactile, current, or torque signal.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
cli = OmegaConf.from_dotlist(sys.argv[1:])
checkpoint = Path(
    str(
        cli.get(
            "checkpoint",
            "artifacts/moving_cube_static_bowl/teachers/"
            "moving_cube_c0u_close_direct_rl_probe20m_4096_no_rtx/"
            "checkpoints/ppo_teacher_14024704.pt",
        )
    )
).resolve()
num_envs = int(cli.get("num_envs", 256))
seed = int(cli.get("seed", 0))
close_steps_option = int(cli.get("close_steps", 10))
lift_steps_option = int(cli.get("lift_steps", 24))
hold_steps_option = int(cli.get("hold_steps", 10))
lift_height_option = float(cli.get("lift_height", 0.008))
# Mean geometry measured at the first grasp indication over 20/20 successful
# randomized 0.10 m/s expert episodes.  The x offset is important: the Franka
# TCP frame origin is not the same point as the centre of a robust pad pinch.
oracle_grasp_offset_values = (
    float(cli.get("oracle_offset_x", 0.0063034)),
    float(cli.get("oracle_offset_y", 0.0000)),
    float(cli.get("oracle_offset_z", -0.0023904)),
)
output = Path(
    str(
        cli.get(
            "output",
            "artifacts/moving_cube_static_bowl/validation/c0y_micro_lift_step1.json",
        )
    )
).resolve()

from pickplace.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=False)

import torch  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from pickplace.metrics import outcome_terms  # noqa: E402
from pickplace.torchrl_env import make_env  # noqa: E402
from pickplace.experts.moving_cube import _topdown_axis_angle_error  # noqa: E402

spec = importlib.util.spec_from_file_location("teacher_utils", HERE / "utils.py")
teacher_utils = importlib.util.module_from_spec(spec)
sys.modules["teacher_utils"] = teacher_utils
spec.loader.exec_module(teacher_utils)

oracle_grasp_offset = torch.tensor(oracle_grasp_offset_values)


def flat(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1)


saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
env_cfg = dict(saved["config"]["env"])
env_cfg.update(num_envs=num_envs, cameras=False, seed=seed, target_lift=0.005)
env_cfg["geometric_close_curriculum"] = {
    "success_steps": 0,
    "distance_threshold": 0.045,
    "min_close_fraction": 0.15,
    "max_close_fraction": 0.90,
    "far_close_penalty": 0.50,
}
env_cfg["geometric_close_lift_bridge"] = {
    "enabled": True,
    "progress_reward": False,
    "close_steps": 5,
    "distance_threshold": 0.045,
    "min_close_fraction": 0.15,
    "max_close_fraction": 0.90,
    "prelift_max_height": 0.003,
    "target_lift": 0.005,
    "max_relative_drift": 0.008,
    "success_steps": 5,
    "tcp_progress_weight": 0.25,
    "object_progress_weight": 1.0,
}
env_cfg["geometric_micro_lift_option"] = {
    "enabled": True,
    "close_steps": close_steps_option,
    "lift_steps": lift_steps_option,
    "hold_steps": hold_steps_option,
    "lift_height": lift_height_option,
    "distance_threshold": 0.045,
    "min_close_fraction": 0.15,
    "max_close_fraction": 0.90,
    "xy_gain": 0.5,
    "z_gain": 1.0,
    "max_xy_step": 0.002,
    "max_z_step": 0.0005,
    "max_finger_delta": 0.00025,
    "force_close": True,
    "freeze_rotation": True,
}
torch.manual_seed(seed)
env = make_env(env_cfg)
actor = teacher_utils.load_teacher_actor(checkpoint, env, env.device)
u = env.base_env._env.unwrapped
max_steps = int(u.max_episode_length) + 1
names = tuple(outcome_terms(env_cfg["task"]))
rest_z = float(u.cfg.belt.plate_top_z + u.cfg.food.item_radius)
open_pos = float(u.cfg.arm.gripper_open)
closed_pos = float(u.cfg.arm.gripper_closed)
finger_span = abs(open_pos - closed_pos)
action_scale = float(u.cfg.arm.ik_action_scale)


class PositionOnlyCloseServo:
    """Current-position servo used only to isolate physical grasp feasibility."""

    def __init__(
        self,
        n: int,
        device: torch.device,
        lead_x: float = 0.0,
        grasp_offset: torch.Tensor | None = None,
    ):
        self.phase = torch.zeros(n, dtype=torch.long, device=device)
        self.phase_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.aligned_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.lead_x = float(lead_x)
        self.grasp_offset = (
            torch.tensor((0.0, 0.0, -0.002), device=device)
            if grasp_offset is None
            else grasp_offset.to(device=device)
        )

    @staticmethod
    def _limited_action(error: torch.Tensor, gain: float = 4.0, limit: float = 0.32) -> torch.Tensor:
        action = gain * error
        norm = torch.linalg.vector_norm(action, dim=-1, keepdim=True)
        return (action * torch.clamp(limit / norm.clamp_min(1.0e-7), max=1.0)).clamp(-1.0, 1.0)

    def act(self, td) -> torch.Tensor:
        history = td.get(("position_history", "position_action"))
        offset = history[:, -1, :3]
        ee = flat(td.get(("proprio", "ee_pos")))
        quat = flat(td.get(("proprio", "ee_quat")))
        n = ee.shape[0]

        desired_offset = torch.zeros_like(offset)
        # Fixed spatial lead in the known conveyor direction.  This is not a
        # measured or inferred speed input and remains identical across the
        # randomized 0.045--0.075 m/s belt range.
        desired_offset[:, 0] = -self.lead_x
        desired_offset[:, 2] = -0.070
        hover_error = offset - desired_offset
        hover_aligned = torch.linalg.vector_norm(hover_error, dim=-1) < 0.008
        self.aligned_steps = torch.where(
            self.phase.eq(0) & hover_aligned,
            self.aligned_steps + 1,
            torch.where(self.phase.eq(0), torch.zeros_like(self.aligned_steps), self.aligned_steps),
        )
        hover_done = self.aligned_steps >= 5
        advance = self.phase.eq(0) & hover_done
        self.phase[advance] = 1
        self.phase_steps[advance] = 0
        self.aligned_steps[advance] = 0

        desired_offset[:] = self.grasp_offset
        descend_error = offset - desired_offset
        descend_aligned = torch.linalg.vector_norm(descend_error, dim=-1) < 0.004
        self.aligned_steps = torch.where(
            self.phase.eq(1) & descend_aligned,
            self.aligned_steps + 1,
            torch.where(self.phase.eq(1), torch.zeros_like(self.aligned_steps), self.aligned_steps),
        )
        descend_done = self.aligned_steps >= 5
        advance = self.phase.eq(1) & descend_done
        self.phase[advance] = 2
        self.phase_steps[advance] = 0
        self.aligned_steps[advance] = 0

        action = torch.zeros(n, 7, device=ee.device)
        action[:, -1] = 1.0
        error = torch.where(self.phase[:, None].eq(0), hover_error, descend_error)
        # Match the validated expert's Cartesian feedback gain but omit its
        # velocity feed-forward entirely.  The target is recomputed from the
        # current image geometry every step.
        action[:, :3] = self._limited_action(error)
        rotation = _topdown_axis_angle_error(quat)
        action[:, 3:6] = (4.0 * rotation).clamp(-1.0, 1.0)
        closing = self.phase.eq(2)
        action[closing, -1] = -1.0
        self.phase_steps += 1
        return action


def evaluate(mode: str) -> dict:
    td = env.reset()
    servo = (
        PositionOnlyCloseServo(num_envs, env.device, grasp_offset=oracle_grasp_offset)
        if mode == "static_position_oracle"
        else None
    )
    if mode == "static_position_oracle":
        # Isolate the 40 mm cube / finger-pad physics from interception.  This
        # is a readout-free static fixture: no velocity is supplied to either
        # the servo or the micro-lift option.
        belt_event = u.event_manager.get_term_cfg("reset_belt").func
        env_ids = torch.arange(num_envs, device=env.device)
        zeros = torch.zeros(num_envs, 1, device=env.device)
        belt_event.speed.zero_()
        belt_event.pallet.write_joint_velocity_to_sim_index(
            velocity=zeros, joint_ids=belt_event.joint_ids, env_ids=env_ids
        )
        belt_event.pallet.set_joint_velocity_target_index(
            target=zeros, joint_ids=belt_event.joint_ids, env_ids=env_ids
        )
        u.scene["food"].write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(num_envs, 6, device=env.device), env_ids=env_ids
        )
    finished = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(finished)
    outcome = ["unfinished"] * num_envs
    ever_near = torch.zeros_like(finished)
    ever_valid_close = torch.zeros_like(finished)
    close_count = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    ever_stable_close = torch.zeros_like(finished)
    max_lift = torch.zeros(num_envs, device=env.device)
    bridge_latched = torch.zeros_like(finished)
    bridge_reference = torch.zeros(num_envs, 3, device=env.device)
    max_coupled_lift = torch.zeros(num_envs, device=env.device)
    lost_distance = torch.zeros_like(finished)
    lost_aperture = torch.zeros_like(finished)
    lost_drift = torch.zeros_like(finished)
    oracle_centered = torch.zeros_like(finished)

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for step in range(max_steps):
            active = ~finished
            history = td.get(("position_history", "position_action"))
            offset = history[:, -1, :3]
            ee = flat(td.get(("proprio", "ee_pos")))
            fingers = flat(td.get(("proprio", "gripper_pos")))
            food_z = ee[:, 2] + offset[:, 2]
            distance = torch.linalg.vector_norm(offset, dim=-1)
            close_fraction = (1.0 - (fingers.mean(-1) - closed_pos) / finger_span).clamp(0.0, 1.0)
            near = distance <= 0.045
            valid_close = near & (close_fraction >= 0.15) & (close_fraction <= 0.90)
            close_count = torch.where(active & valid_close, close_count + 1, torch.zeros_like(close_count))
            ever_near |= active & near
            ever_valid_close |= active & valid_close
            ever_stable_close |= active & (close_count >= 5)
            newly_latched = active & ~bridge_latched & (close_count >= 5)
            bridge_reference = torch.where(newly_latched[:, None], offset, bridge_reference)
            bridge_latched |= newly_latched
            lift = (food_z - rest_z).clamp_min(0.0)
            drift = torch.linalg.vector_norm(offset - bridge_reference, dim=-1)
            coupled = valid_close & (drift <= 0.008)
            max_coupled_lift = torch.where(
                active & bridge_latched & coupled,
                torch.maximum(max_coupled_lift, lift),
                max_coupled_lift,
            )
            lost = active & bridge_latched & ~coupled
            lost_distance |= lost & ~near
            aperture_ok = (close_fraction >= 0.15) & (close_fraction <= 0.90)
            lost_aperture |= lost & ~aperture_ok
            lost_drift |= lost & near & aperture_ok & (drift > 0.008)
            bridge_latched &= ~lost
            close_count = torch.where(lost, torch.zeros_like(close_count), close_count)
            max_lift = torch.where(active, torch.maximum(max_lift, lift), max_lift)

            if mode == "c0u_gate":
                td = actor(td)
            else:
                oracle_action = servo.act(td)
                newly_closing = servo.phase.eq(2) & ~oracle_centered & active
                if bool(newly_closing.any()):
                    ids = newly_closing.nonzero(as_tuple=False).flatten()
                    root_pose = torch.zeros(ids.numel(), 7, device=env.device)
                    calibrated_position = ee[ids] + oracle_grasp_offset.to(env.device)
                    root_pose[:, :2] = calibrated_position[:, :2] + u.scene.env_origins[ids, :2]
                    # Keep the fixture resting on the belt.  The measured z
                    # offset is used by the servo, while this avoids spawning
                    # the cube through the support surface due to sub-mm IK
                    # tracking error.
                    root_pose[:, 2] = rest_z + u.scene.env_origins[ids, 2]
                    root_pose[:, 6] = 1.0
                    u.scene["food"].write_root_pose_to_sim_index(root_pose=root_pose, env_ids=ids)
                    u.scene["food"].write_root_velocity_to_sim_index(
                        root_velocity=torch.zeros(ids.numel(), 6, device=env.device), env_ids=ids
                    )
                    oracle_centered[ids] = True
                td.set("action", oracle_action)
            stepped, td = env.step_and_maybe_reset(td)
            done = flat(stepped.get(("next", "done")))[:, 0].bool()
            newly_done = active & done
            for name in names:
                fired = flat(stepped.get(("next", "outcome", name)))[:, 0].bool()
                for index in (newly_done & fired).nonzero(as_tuple=False).flatten().tolist():
                    outcome[index] = name
                if name == "success":
                    success |= newly_done & fired
            finished |= newly_done
            if bool(finished.all()):
                break

    controllers = [
        transform.controller
        for transform in env.transform.transforms
        if hasattr(transform, "controller") and transform.controller.__class__.__name__ == "GeometricMicroLiftOption"
    ]
    option_controller = controllers[0] if controllers else None
    option_phase = option_controller.phase if option_controller is not None else None
    return {
        "mode": mode,
        "steps": step + 1,
        "successes": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "outcomes": dict(sorted(Counter(outcome).items())),
        "ever_near_rate": float(ever_near.float().mean().item()),
        "ever_valid_close_rate": float(ever_valid_close.float().mean().item()),
        "ever_stable_close_rate": float(ever_stable_close.float().mean().item()),
        "lift_1mm_rate": float((max_lift >= 0.001).float().mean().item()),
        "lift_3mm_rate": float((max_lift >= 0.003).float().mean().item()),
        "lift_5mm_rate": float((max_lift >= 0.005).float().mean().item()),
        "mean_max_lift_m": float(max_lift.mean().item()),
        "coupled_lift_1mm_rate": float((max_coupled_lift >= 0.001).float().mean().item()),
        "coupled_lift_3mm_rate": float((max_coupled_lift >= 0.003).float().mean().item()),
        "coupled_lift_5mm_rate": float((max_coupled_lift >= 0.005).float().mean().item()),
        "mean_max_coupled_lift_m": float(max_coupled_lift.mean().item()),
        "loss_rates": {
            "distance": float(lost_distance.float().mean().item()),
            "aperture": float(lost_aperture.float().mean().item()),
            "relative_drift": float(lost_drift.float().mean().item()),
        },
        "option_telemetry": {
            "controller_count": len(controllers),
            "triggered_rate": (
                float(option_phase.gt(0).float().mean().item()) if option_phase is not None else 0.0
            ),
            "finished_rate": (
                float(option_phase.gt(lift_steps_option + hold_steps_option).float().mean().item())
                if option_phase is not None else 0.0
            ),
            "max_close_count": (
                int(option_controller.close_count.max().item()) if option_controller is not None else 0
            ),
        },
        "oracle_centered_rate": float(oracle_centered.float().mean().item()),
    }


report = {
    "checkpoint": str(checkpoint),
    "num_envs": num_envs,
    "seed": seed,
    "cube_size_m": float(u.cfg.food.size),
    "cube_mass_kg": float(u.cfg.food.mass),
    "speed_free_contract": {
        "object_velocity": False,
        "contact": False,
        "force": False,
        "tactile": False,
        "motor_current": False,
        "inputs": ["object_to_tcp_position_history", "tcp_pose", "finger_encoders"],
    },
    "micro_lift_option": env_cfg["geometric_micro_lift_option"],
    "static_oracle_grasp_offset_object_minus_tcp_m": oracle_grasp_offset.tolist(),
    "tests": [evaluate("static_position_oracle"), evaluate("c0u_gate")],
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("C0Y_STEP1 " + json.dumps(report, ensure_ascii=False), flush=True)
os._exit(0)
