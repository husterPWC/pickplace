"""Trace deterministic G0 sphere PPO policies around the grasp-option gates.

This diagnostic is deliberately read-only: it loads complete checkpoints,
runs one deterministic episode per environment, and reports how closely the
policy approached the 6D trigger basin.  Contact/force are not policy inputs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from pickplace.app import launch_app


parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+", type=Path)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--canonical-task",
    action="store_true",
    help="Evaluate a compatible historical actor with the canonical sphere observation/task.",
)
args = parser.parse_args()

app = launch_app(headless=True, enable_cameras=False)

import torch  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from pickplace.torchrl_env import make_env  # noqa: E402


def load_teacher_utils():
    path = Path(__file__).resolve().parents[1] / "pipeline/0_state_teacher/utils.py"
    spec = importlib.util.spec_from_file_location("sphere_g0_ppo_diagnostic_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def quantiles(value: torch.Tensor) -> dict[str, float]:
    q = torch.quantile(value.float(), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=value.device))
    return {name: float(item) for name, item in zip(("min", "q25", "median", "q75", "max"), q)}


def main() -> None:
    states = [torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoints]
    env_cfg = dict(states[0]["config"]["env"])
    env_cfg.update(num_envs=args.num_envs, cameras=False, seed=args.seed)
    if args.canonical_task:
        env_cfg["task"] = "FoodRobot-MovingSphereCanonicalGapLift-v0"
    torch.manual_seed(args.seed)
    env = make_env(env_cfg)
    teacher_utils = load_teacher_utils()
    option = next(
        transform.controller
        for transform in env.transform.transforms
        if hasattr(transform, "controller")
        and transform.controller.__class__.__name__ == "StableSphereCloseHoldLiftOption"
    )
    cfg = option.config
    reports = []

    for path, state in zip(args.checkpoints, states):
        actor = teacher_utils.load_teacher_actor(path, env, env.device)
        actor.eval()
        td = env.reset()
        n = args.num_envs
        finished = torch.zeros(n, dtype=torch.bool, device=env.device)
        success = torch.zeros_like(finished)
        exited = torch.zeros_like(finished)
        min_raw_translation = torch.full((n,), float("inf"), device=env.device)
        min_raw_rotation = torch.full((n,), float("inf"), device=env.device)
        min_filtered_translation = torch.full((n,), float("inf"), device=env.device)
        min_filtered_rotation = torch.full((n,), float("inf"), device=env.device)
        ever_raw_joint_aligned = torch.zeros_like(finished)
        ever_filtered_joint_aligned = torch.zeros_like(finished)
        max_phase = torch.zeros(n, dtype=torch.long, device=env.device)
        max_alignment_count = torch.zeros_like(max_phase)
        action_abs_sum = torch.zeros(7, device=env.device)
        action_samples = 0
        max_steps = int(env.base_env._env.unwrapped.max_episode_length) + 1

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for step in range(max_steps):
                live = ~finished
                td = actor(td)
                policy_action = td["action"].clone()
                action_abs_sum += policy_action[live].abs().sum(0)
                action_samples += int(live.sum())

                raw = td["grasp_guidance", "best_pose_delta"]
                raw_translation = torch.linalg.vector_norm(raw[:, :3], dim=-1)
                raw_rotation = torch.linalg.vector_norm(raw[:, 3:], dim=-1)
                min_raw_translation[live] = torch.minimum(
                    min_raw_translation[live], raw_translation[live]
                )
                min_raw_rotation[live] = torch.minimum(min_raw_rotation[live], raw_rotation[live])
                ever_raw_joint_aligned |= live & (raw_translation <= cfg.trigger_translation) & (
                    raw_rotation <= cfg.rotation_threshold
                )
                ee_pos = td["proprio", "ee_pos"].clone()
                transition, td = env.step_and_maybe_reset(td)

                if option.target.initialized is not None:
                    filtered_translation = torch.linalg.vector_norm(
                        option.target.position - ee_pos, dim=-1
                    )
                    filtered_rotation = torch.linalg.vector_norm(option.target.rotation_delta, dim=-1)
                    min_filtered_translation[live] = torch.minimum(
                        min_filtered_translation[live], filtered_translation[live]
                    )
                    min_filtered_rotation[live] = torch.minimum(
                        min_filtered_rotation[live], filtered_rotation[live]
                    )
                    ever_filtered_joint_aligned |= live & (
                        filtered_translation <= cfg.trigger_translation
                    ) & (filtered_rotation <= cfg.rotation_threshold)
                if option.phase is not None:
                    max_phase[live] = torch.maximum(max_phase[live], option.phase[live])
                    max_alignment_count[live] = torch.maximum(
                        max_alignment_count[live], option.alignment_count[live]
                    )

                done = transition["next", "done"].reshape(-1) & live
                if done.any():
                    success |= transition["next", "outcome", "success"].reshape(-1) & done
                    exited |= transition["next", "outcome", "cube_exited_zone"].reshape(-1) & done
                    finished |= done
                if finished.all():
                    break

        phase_counts = {
            option.PHASE_NAMES[index]: int((max_phase == index).sum())
            for index in range(len(option.PHASE_NAMES))
        }
        reports.append(
            {
                "checkpoint": str(path),
                "frames": int(state["frames"]),
                "environment_task": env_cfg["task"],
                "episodes": n,
                "steps": step + 1,
                "successes": int(success.sum()),
                "exited_zone": int(exited.sum()),
                "raw_joint_trigger_aligned": int(ever_raw_joint_aligned.sum()),
                "filtered_joint_trigger_aligned": int(ever_filtered_joint_aligned.sum()),
                "option_started": int((max_phase > option.IDLE).sum()),
                "option_phase_counts": phase_counts,
                "max_alignment_count": int(max_alignment_count.max()),
                "min_raw_translation_m": quantiles(min_raw_translation),
                "min_raw_rotation_rad": quantiles(min_raw_rotation),
                "min_filtered_translation_m": quantiles(min_filtered_translation),
                "min_filtered_rotation_rad": quantiles(min_filtered_rotation),
                "mean_abs_policy_action": (action_abs_sum / max(action_samples, 1)).tolist(),
                "gate": {
                    "translation_m": cfg.trigger_translation,
                    "rotation_rad": cfg.rotation_threshold,
                    "consecutive_steps": cfg.alignment_steps,
                    "grasp_translation_m": cfg.grasp_translation,
                },
            }
        )

    print("SPHERE_G0_PPO_DIAGNOSIS " + json.dumps(reports), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
