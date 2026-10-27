"""Trace the early fixed-line sphere interceptor inside Isaac.

This is a read-only diagnostic.  It keeps the actor observation contract
unchanged and records simulator state only after the policy/controller have
acted, so privileged values never become policy inputs.
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
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--speed", type=float, default=0.05)
parser.add_argument("--pallet-start", type=float, default=0.0)
parser.add_argument("--pallet-start-min", type=float)
parser.add_argument("--pallet-start-max", type=float)
parser.add_argument("--food-offset-x-min", type=float, default=0.0)
parser.add_argument("--food-offset-x-max", type=float, default=0.0)
parser.add_argument("--food-offset-y-min", type=float, default=0.0)
parser.add_argument("--food-offset-y-max", type=float, default=0.0)
parser.add_argument("--visual-upstream-extension", type=float, default=0.0)
parser.add_argument("--episode-completion-margin", type=float, default=2.0)
parser.add_argument("--lead", type=float, default=0.16)
parser.add_argument("--align-lead", type=float, default=0.04)
parser.add_argument("--pre-descent-height", type=float, default=0.02)
parser.add_argument("--lift-height", type=float, default=0.05)
parser.add_argument("--lift-steps", type=int, default=60)
parser.add_argument("--lift-timeout-steps", type=int, default=150)
parser.add_argument(
    "--lift-target-slack",
    type=float,
    default=0.0,
    help="Positive geometric Z slack; <=0 disables the optional clamp.",
)
parser.add_argument(
    "--lift-detach-height",
    type=float,
    default=0.0,
    help="Positive open-loop peel height before follower limiting; <=0 disables it.",
)
parser.add_argument("--lift-xy-gain", type=float, default=0.5)
parser.add_argument("--max-lift-xy-step", type=float, default=0.002)
parser.add_argument("--target-xy-feedforward-gain", type=float, default=0.0)
parser.add_argument("--max-target-xy-feedforward-step", type=float, default=0.0)
parser.add_argument(
    "--align-target-xy-feedforward",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--close-requires-raw-alignment",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument("--max-close-hold-xy-step", type=float, default=0.0)
parser.add_argument(
    "--latch-lift-xy-feedforward",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument("--lift-geometry-relatch-height", type=float, default=0.0)
parser.add_argument(
    "--lift-relative-z-gain",
    type=float,
    default=0.0,
    help="Positive geometry-only relative-Z correction gain; <=0 disables it.",
)
parser.add_argument("--close-action", type=float, default=-0.15)
parser.add_argument(
    "--preserve-lift-relative-xy",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--settled-coupling-reference",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument("--translation-residual-limit", type=float, default=0.0)
parser.add_argument("--rotation-residual-limit", type=float, default=0.0)
parser.add_argument("--trace-stride", type=int, default=10)
parser.add_argument("--max-steps", type=int)
parser.add_argument("--output", type=Path)
args = parser.parse_args()

app = launch_app(headless=True, enable_cameras=False)

import torch  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from pickplace.torchrl_env import make_env  # noqa: E402


def _teacher_utils():
    path = Path(__file__).resolve().parents[1] / "pipeline/0_state_teacher/utils.py"
    spec = importlib.util.spec_from_file_location("early_intercept_diagnostic_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _vec(value: torch.Tensor) -> list[float]:
    return [round(float(item), 6) for item in value.detach().cpu().reshape(-1)]


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    levels = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=value.device)
    values = torch.quantile(value.float(), levels)
    return {
        name: round(float(item), 6)
        for name, item in zip(("min", "q25", "median", "q75", "max"), values)
    }


def main() -> None:
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    pallet_start_range = [
        args.pallet_start if args.pallet_start_min is None else args.pallet_start_min,
        args.pallet_start if args.pallet_start_max is None else args.pallet_start_max,
    ]
    food_offset_x = [args.food_offset_x_min, args.food_offset_x_max]
    food_offset_y = [args.food_offset_y_min, args.food_offset_y_max]
    overrides = {
        "robot_initial_joint_pos": {
            "panda_joint1": -1.55131948,
            "panda_joint2": -0.25804791,
            "panda_joint3": 2.25832868,
            "panda_joint4": -2.45404768,
            "panda_joint5": 0.38310212,
            "panda_joint6": 2.59809756,
            "panda_joint7": 1.02313423,
        },
        "belt": {
            "speed": args.speed,
            "speed_noise": 0.0,
            "place_window": 0.40 / args.speed,
            "pallet_start_range": pallet_start_range,
            "bowl_offset_x": food_offset_x,
            "bowl_offset_y": food_offset_y,
            "visual_upstream_extension": args.visual_upstream_extension,
            "episode_completion_margin": args.episode_completion_margin,
            "pallet": {"travel_lower": min(-0.12, pallet_start_range[0] - 0.04)},
        },
        "stable_sphere_residual_skill": {
            "intercept_lead_distance": args.lead,
            "intercept_align_lead_distance": args.align_lead,
            "pre_descent_height": args.pre_descent_height,
            "lift_height": args.lift_height,
            "lift_steps": args.lift_steps,
            "lift_timeout_steps": args.lift_timeout_steps,
            "lift_target_slack": (
                None if args.lift_target_slack <= 0.0 else args.lift_target_slack
            ),
            "lift_detach_height": (
                None if args.lift_detach_height <= 0.0 else args.lift_detach_height
            ),
            "lift_xy_gain": args.lift_xy_gain,
            "max_lift_xy_step": args.max_lift_xy_step,
            "target_xy_feedforward_gain": args.target_xy_feedforward_gain,
            "max_target_xy_feedforward_step": (
                None
                if args.max_target_xy_feedforward_step <= 0.0
                else args.max_target_xy_feedforward_step
            ),
            "align_target_xy_feedforward": args.align_target_xy_feedforward,
            "close_requires_raw_alignment": args.close_requires_raw_alignment,
            "max_close_hold_xy_step": (
                None if args.max_close_hold_xy_step <= 0.0 else args.max_close_hold_xy_step
            ),
            "latch_lift_xy_feedforward": args.latch_lift_xy_feedforward,
            "lift_geometry_relatch_height": (
                None
                if args.lift_geometry_relatch_height <= 0.0
                else args.lift_geometry_relatch_height
            ),
            "lift_relative_z_gain": (
                None if args.lift_relative_z_gain <= 0.0 else args.lift_relative_z_gain
            ),
            "preserve_lift_relative_xy": args.preserve_lift_relative_xy,
            "settled_coupling_reference": args.settled_coupling_reference,
            "close_action": args.close_action,
            "translation_residual_limit": args.translation_residual_limit,
            "rotation_residual_limit": args.rotation_residual_limit,
        },
    }
    env_cfg = OmegaConf.to_container(
        OmegaConf.merge(state["config"]["env"], overrides), resolve=True
    )
    env_cfg.update(num_envs=args.num_envs, cameras=False, seed=args.seed)
    torch.manual_seed(args.seed)
    env = make_env(env_cfg)
    actor = _teacher_utils().load_teacher_actor(args.checkpoint, env, env.device)
    actor.eval()

    residual = next(
        transform.controller
        for transform in env.transform.transforms
        if hasattr(transform, "controller")
        and transform.controller.__class__.__name__ == "StableSphereResidualSkill"
    )
    skill = residual.baseline
    base = env.base_env._env.unwrapped
    scene = base.scene
    escape_term = base.termination_manager.get_term_cfg("tcp_escape").func
    success_term = base.termination_manager.get_term_cfg("success").func

    td = env.reset()
    n = args.num_envs
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(finished)
    outcome_names = ("cube_exited_zone", "cube_off_belt", "tcp_escape", "time_out")
    outcomes = {name: torch.zeros_like(finished) for name in outcome_names}
    min_raw_translation = torch.full((n,), float("inf"), device=env.device)
    min_filtered_translation = torch.full((n,), float("inf"), device=env.device)
    tcp_min = torch.full((n, 3), float("inf"), device=env.device)
    tcp_max = torch.full((n, 3), -float("inf"), device=env.device)
    max_phase = torch.zeros(n, dtype=torch.long, device=env.device)
    max_alignment_count = torch.zeros(n, dtype=torch.long, device=env.device)
    max_retreat_count = torch.zeros(n, dtype=torch.long, device=env.device)
    max_true_lift = torch.zeros(n, device=env.device)
    ever_dual_contact = torch.zeros(n, dtype=torch.bool, device=env.device)
    ever_coupled = torch.zeros(n, dtype=torch.bool, device=env.device)
    max_coupling_hold_count = torch.zeros(n, dtype=torch.long, device=env.device)
    max_success_count = torch.zeros(n, dtype=torch.long, device=env.device)
    timeline: list[dict[str, object]] = []
    prior_phase = -1
    max_steps = args.max_steps or (int(base.max_episode_length) + 1)
    initial_food_z = (scene["food"].data.root_pos_w.torch - scene.env_origins)[:, 2].clone()

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for step in range(max_steps):
            live = ~finished
            ee = td["proprio", "ee_pos"].clone()
            raw = td["grasp_guidance", "best_pose_delta"].clone()
            raw_translation = torch.linalg.vector_norm(raw[:, :3], dim=-1)
            raw_rotation = torch.linalg.vector_norm(raw[:, 3:], dim=-1)
            min_raw_translation[live] = torch.minimum(
                min_raw_translation[live], raw_translation[live]
            )
            tcp_min[live] = torch.minimum(tcp_min[live], ee[live])
            tcp_max[live] = torch.maximum(tcp_max[live], ee[live])
            food = scene["food"].data.root_pos_w.torch - scene.env_origins
            max_true_lift[live] = torch.maximum(
                max_true_lift[live], food[live, 2] - initial_food_z[live]
            )
            left_force = scene["left_finger_contact"].data.force_matrix_w.norm(dim=-1).amax(
                dim=(-1, -2)
            )
            right_force = scene["right_finger_contact"].data.force_matrix_w.norm(dim=-1).amax(
                dim=(-1, -2)
            )
            ever_dual_contact |= live & (left_force > 0.05) & (right_force > 0.05)
            ever_coupled |= live & success_term.coupled
            max_coupling_hold_count[live] = torch.maximum(
                max_coupling_hold_count[live], success_term.hold_count[live]
            )
            max_success_count[live] = torch.maximum(
                max_success_count[live], success_term.success_count[live]
            )

            phase = skill.phase.clone() if skill.phase is not None else torch.zeros(
                n, dtype=torch.long, device=env.device
            )
            alignment = (
                skill.alignment_count.clone()
                if skill.alignment_count is not None
                else torch.zeros(n, dtype=torch.long, device=env.device)
            )
            retreat = escape_term.retreat_count.clone()
            max_phase[live] = torch.maximum(max_phase[live], phase[live])
            max_alignment_count[live] = torch.maximum(
                max_alignment_count[live], alignment[live]
            )
            max_retreat_count[live] = torch.maximum(max_retreat_count[live], retreat[live])

            filtered_position = None
            filtered_distance = None
            if skill.target.position is not None:
                filtered_position = skill.target.position.clone()
                filtered_distance = torch.linalg.vector_norm(filtered_position - ee, dim=-1)
                min_filtered_translation[live] = torch.minimum(
                    min_filtered_translation[live], filtered_distance[live]
                )

            phase0 = int(phase[0])
            if step % args.trace_stride == 0 or phase0 != prior_phase:
                row = {
                    "step": step,
                    "time_s": round(step * float(base.step_dt), 4),
                    "phase": skill.PHASE_NAMES[phase0],
                    "tcp_xyz": _vec(ee[0]),
                    "tcp_quaternion_xyzw": _vec(td["proprio", "ee_quat"][0]),
                    "sphere_xyz": _vec(food[0]),
                    "raw_grasp_xyz": _vec(ee[0] + raw[0, :3]),
                    "raw_translation_m": round(float(raw_translation[0]), 6),
                    "raw_rotation_rad": round(float(raw_rotation[0]), 6),
                    "raw_rotation_vector": _vec(raw[0, 3:]),
                    "alignment_count": int(alignment[0]),
                    "retreat_count": int(retreat[0]),
                    "dual_contact": bool((left_force[0] > 0.05) & (right_force[0] > 0.05)),
                    "coupled": bool(success_term.coupled[0]),
                    "coupling_hold_count": int(success_term.hold_count[0]),
                    "success_count": int(success_term.success_count[0]),
                }
                if filtered_position is not None:
                    row["filtered_grasp_xyz"] = _vec(filtered_position[0])
                    row["filtered_translation_m"] = round(float(filtered_distance[0]), 6)
                    row["filtered_rotation_rad"] = round(
                        float(torch.linalg.vector_norm(skill.target.rotation_delta[0])), 6
                    )
                    row["target_stable_count"] = int(skill.target.stable_count[0])
                    row["selected_candidate"] = int(skill.target.selected_index[0])
                timeline.append(row)
            prior_phase = phase0

            td = actor(td)
            transition, td = env.step_and_maybe_reset(td)
            done = transition["next", "done"].reshape(-1) & live
            if done.any():
                success |= transition["next", "outcome", "success"].reshape(-1) & done
                for name in outcome_names:
                    outcomes[name] |= transition["next", "outcome", name].reshape(-1) & done
                finished |= done
            if finished.all():
                break

    final_live_tcp = tcp_max[:, 0]  # forces finite diagnostics if an episode ended immediately
    del final_live_tcp
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_frames": int(state["frames"]),
        "episodes": n,
        "steps": step + 1,
        "speed_mps": args.speed,
        "pallet_start_joint_range_m": pallet_start_range,
        "food_offset_x_range_m": food_offset_x,
        "food_offset_y_range_m": food_offset_y,
        "sphere_start_x_range_m": [
            0.20 + pallet_start_range[0] + food_offset_x[0],
            0.20 + pallet_start_range[1] + food_offset_x[1],
        ],
        "visual_upstream_extension_m": args.visual_upstream_extension,
        "episode_completion_margin_s": args.episode_completion_margin,
        "interceptor": {
            "intercept_x_m": skill.config.intercept_x,
            "lead_m": args.lead,
            "align_lead_m": args.align_lead,
            "pre_descent_height_m": args.pre_descent_height,
            "grasp_translation_threshold_m": skill.config.grasp_translation_threshold,
        },
        "residual_limits": {
            "translation": args.translation_residual_limit,
            "rotation": args.rotation_residual_limit,
        },
        "alignment_following": {
            "enabled": args.align_target_xy_feedforward,
            "target_xy_feedforward_gain": args.target_xy_feedforward_gain,
            "close_requires_raw_alignment": args.close_requires_raw_alignment,
            "max_target_xy_feedforward_step_m": (
                None
                if args.max_target_xy_feedforward_step <= 0.0
                else args.max_target_xy_feedforward_step
            ),
            "max_close_hold_xy_step_m": (
                None if args.max_close_hold_xy_step <= 0.0 else args.max_close_hold_xy_step
            ),
        },
        "successes": int(success.sum()),
        "outcomes": {name: int(mask.sum()) for name, mask in outcomes.items()},
        "phase_reached": {
            name: int((max_phase >= index).sum())
            for index, name in enumerate(skill.PHASE_NAMES)
        },
        "max_alignment_count": int(max_alignment_count.max()),
        "max_retreat_count": int(max_retreat_count.max()),
        "true_lift_m": _quantiles(max_true_lift),
        "ever_dual_contact": int(ever_dual_contact.sum()),
        "ever_coupled": int(ever_coupled.sum()),
        "max_coupling_hold_count": int(max_coupling_hold_count.max()),
        "max_success_count": int(max_success_count.max()),
        "min_raw_translation_m": _quantiles(min_raw_translation),
        "min_filtered_translation_m": _quantiles(min_filtered_translation),
        "tcp_min_xyz_m": [_quantiles(tcp_min[:, axis]) for axis in range(3)],
        "tcp_max_xyz_m": [_quantiles(tcp_max[:, axis]) for axis in range(3)],
        "workspace_lower_m": [0.10, -0.20, 0.02],
        "workspace_upper_m": [0.75, 0.48, 0.75],
        "env0_timeline": timeline,
        "diagnostic_only_privileged_values": ["sphere_xyz", "termination_counters"],
    }
    output = args.output or Path(
        "artifacts/moving_sphere_static_bowl/diagnostics/"
        "g0_s1c_early_intercept_speed_005_zero_residual.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("EARLY_INTERCEPT_DIAGNOSIS " + json.dumps(report), flush=True)
    print(f"WROTE {output.resolve()}", flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
