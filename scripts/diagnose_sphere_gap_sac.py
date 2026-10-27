"""Evaluate sphere-gap SAC checkpoints and expose policy/critic collapse signals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pickplace.app import launch_app


parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+", type=Path)
parser.add_argument("--num-envs", type=int, default=8)
args = parser.parse_args()

app = launch_app(headless=True, enable_cameras=False, device="cuda:0")

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pickplace.torchrl_env import make_env  # noqa: E402

import importlib.util  # noqa: E402


def load_utils():
    path = Path(__file__).resolve().parents[1] / "pipeline/0_state_teacher/sac_utils.py"
    spec = importlib.util.spec_from_file_location("sphere_gap_sac_utils_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    states = [torch.load(path, map_location="cpu", weights_only=False) for path in args.checkpoints]
    env_cfg = dict(states[0]["config"]["env"])
    env_cfg.update(num_envs=args.num_envs, device="cuda:0")
    env = make_env(env_cfg)
    obs_keys = [tuple(key) for key in states[0]["config"]["network"]["in_keys"]]
    grasp_key = tuple(states[0]["config"]["network"]["grasp_in_key"])
    spec = env.observation_spec
    batch_ndim = len(env.batch_size)
    shapes = [tuple(spec[key].shape[batch_ndim:]) for key in obs_keys]
    su = load_utils()
    net_cfg = OmegaConf.create(states[0]["config"]["network"])
    actor, q1, q2, _, _ = su.make_sac_models(
        shapes, obs_keys.index(grasp_key), int(env.action_spec.shape[-1]), net_cfg, torch.device("cuda:0")
    )
    option = next(
        transform.controller
        for transform in env.transform.transforms
        if hasattr(transform, "controller")
        and transform.controller.__class__.__name__ == "GraspPoseCloseRetractOption"
    )

    reports = []
    for path, state in zip(args.checkpoints, states):
        actor.load_state_dict(state["actor"])
        q1.load_state_dict(state["q1"])
        q2.load_state_dict(state["q2"])
        actor.eval(); q1.eval(); q2.eval()
        td = env.reset()
        finished = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        success = torch.zeros_like(finished)
        outcomes = {
            name: torch.zeros_like(finished)
            for name in ("cube_exited_zone", "cube_off_belt", "time_out")
        }
        option_triggered = torch.zeros_like(finished)
        min_pose_error = torch.full((args.num_envs,), float("inf"), device=env.device)
        accum = {
            "samples": 0,
            "reward": 0.0,
            "actor_active": 0.0,
            "q1_active": 0.0,
            "q2_active": 0.0,
            "q1": 0.0,
            "q2": 0.0,
            "log_std": 0.0,
        }
        action_sum = torch.zeros(7, device=env.device)
        action_abs_sum = torch.zeros(7, device=env.device)
        max_steps = int(env.base_env._env.unwrapped.max_episode_length)
        for step in range(max_steps):
            obs = [td.get(key) for key in obs_keys]
            with torch.no_grad():
                mean, log_std = actor(*obs)
                action = torch.tanh(mean)
                actor_latent = actor.encoder(*obs)
                q1_latent = q1.encoder(*obs)
                q2_latent = q2.encoder(*obs)
                q1_value = q1(*obs, action)
                q2_value = q2(*obs, action)
            live = ~finished
            count = int(live.sum())
            if count:
                accum["samples"] += count
                accum["actor_active"] += float((actor_latent[live] > 0).float().sum())
                accum["q1_active"] += float((q1_latent[live] > 0).float().sum())
                accum["q2_active"] += float((q2_latent[live] > 0).float().sum())
                accum["q1"] += float(q1_value[live].sum())
                accum["q2"] += float(q2_value[live].sum())
                accum["log_std"] += float(log_std[live].sum())
                action_sum += action[live].sum(0)
                action_abs_sum += action[live].abs().sum(0)
            pose_error = td["grasp_guidance", "best_pose_delta"][:, :3].norm(dim=-1)
            min_pose_error = torch.minimum(min_pose_error, pose_error)
            td.set("action", action)
            transition, td = env.step_and_maybe_reset(td)
            if option.phase is not None:
                option_triggered |= option.phase > 0
            accum["reward"] += float(transition["next", "reward"][live].sum())
            done = transition["next", "done"].reshape(-1) & live
            if done.any():
                success |= transition["next", "outcome", "success"].reshape(-1) & done
                for name in outcomes:
                    outcomes[name] |= transition["next", "outcome", name].reshape(-1) & done
                finished |= done
            if finished.all():
                break
        samples = max(accum["samples"], 1)
        latent_values = samples * int(net_cfg.latent_dim)
        reports.append(
            {
                "checkpoint": str(path),
                "frames": int(state["frames"]),
                "alpha": float(state["log_alpha"].exp()),
                "episodes": args.num_envs,
                "successes": int(success.sum()),
                **{name: int(mask.sum()) for name, mask in outcomes.items()},
                "option_triggered": int(option_triggered.sum()),
                "min_pose_error_mean": float(min_pose_error.mean()),
                "mean_reward_per_step": accum["reward"] / samples,
                "actor_latent_active_fraction": accum["actor_active"] / latent_values,
                "q1_latent_active_fraction": accum["q1_active"] / latent_values,
                "q2_latent_active_fraction": accum["q2_active"] / latent_values,
                "q1_mean": accum["q1"] / samples,
                "q2_mean": accum["q2"] / samples,
                "log_std_mean": accum["log_std"] / (samples * 7),
                "action_mean": (action_sum / samples).tolist(),
                "action_abs_mean": (action_abs_sum / samples).tolist(),
                "steps": step + 1,
            }
        )
    print("SPHERE_GAP_SAC_DIAGNOSIS " + json.dumps(reports), flush=True)
    env.close()
    os._exit(0)


if __name__ == "__main__":
    main()
