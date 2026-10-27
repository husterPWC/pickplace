"""Off-policy SAC trainer for the independent speed-free sphere_gap task."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

HERE = Path(__file__).resolve().parent


def _local_utils():
    spec = importlib.util.spec_from_file_location("sphere_gap_sac_utils", HERE / "sac_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@hydra.main(config_path="", config_name="config_sphere_gap_s0a", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.env.cameras:
        raise ValueError("sphere_gap SAC is a state/geometric teacher; set env.cameras=false")
    if cfg.env.task != "FoodRobot-MovingSphereGapLift-v0":
        raise ValueError("sac_train.py is intentionally scoped to FoodRobot-MovingSphereGapLift-v0")

    from pickplace.app import launch_app

    launch_app(headless=cfg.app.headless, enable_cameras=False, device=cfg.env.device)

    import torch

    from pickplace.artifacts import git_commit, new_run_dir, sha256_file, update_json, write_json
    from pickplace.torchrl_env import make_env

    su = _local_utils()
    device = torch.device(cfg.env.device)
    torch.manual_seed(int(cfg.env.seed))
    torch.set_float32_matmul_precision(str(cfg.optim.matmul_precision))
    env = make_env(OmegaConf.to_container(cfg.env, resolve=True))
    obs_keys = [tuple(key) for key in cfg.network.in_keys]
    grasp_key = tuple(cfg.network.grasp_in_key)
    if grasp_key not in obs_keys:
        raise ValueError("network.grasp_in_key must be present in network.in_keys")
    spec = env.observation_spec
    batch_ndim = len(env.batch_size)
    shapes = [tuple(spec[key].shape[batch_ndim:]) for key in obs_keys]
    action_dim = int(env.action_spec.shape[-1])
    if action_dim != 7:
        raise ValueError(f"sphere_gap requires 6D EE + 1 continuous gripper action, got {action_dim}")

    actor, q1, q2, target_q1, target_q2 = su.make_sac_models(
        shapes, obs_keys.index(grasp_key), action_dim, cfg.network, device
    )
    actor_optim = torch.optim.Adam(actor.parameters(), lr=float(cfg.sac.actor_lr))
    q_optim = torch.optim.Adam(
        list(q1.parameters()) + list(q2.parameters()), lr=float(cfg.sac.critic_lr)
    )
    log_alpha = torch.tensor(
        float(cfg.sac.log_alpha_init), dtype=torch.float32, device=device, requires_grad=True
    )
    alpha_optim = torch.optim.Adam([log_alpha], lr=float(cfg.sac.alpha_lr))
    target_entropy = float(cfg.sac.target_entropy or -action_dim)

    if cfg.init_checkpoint:
        state = torch.load(cfg.init_checkpoint, map_location=device, weights_only=False)
        required = {"actor", "q1", "q2"}
        if missing := required.difference(state):
            raise ValueError(f"init_checkpoint is missing {sorted(missing)}")
        actor.load_state_dict(state["actor"])
        q1.load_state_dict(state["q1"])
        q2.load_state_dict(state["q2"])
        target_q1.load_state_dict(state.get("target_q1", state["q1"]))
        target_q2.load_state_dict(state.get("target_q2", state["q2"]))
        if "log_alpha" in state:
            log_alpha.data.copy_(torch.as_tensor(state["log_alpha"], device=device))

    replay = su.ReplayBuffer(
        int(cfg.replay.capacity), shapes, grasp_index=obs_keys.index(grasp_key)
    )
    run_dir = Path(cfg.run.dir) if cfg.run.dir else new_run_dir("sphere_gap", cfg.run.name)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_plain = OmegaConf.to_container(cfg, resolve=True)
    manifest_path = run_dir / "manifest.json"
    write_json(
        manifest_path,
        {
            "kind": "sphere_gap_sac",
            "git_commit": git_commit(),
            "config": config_plain,
            "init_checkpoint": cfg.init_checkpoint or None,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ended_at": None,
            "frames": 0,
            "checkpoints": [],
            "stop_reason": None,
        },
    )

    actor.train()
    total_frames = int(cfg.collector.total_frames)
    num_envs = int(env.batch_size[0])
    if int(cfg.replay.learning_starts) < int(cfg.replay.batch_size):
        raise ValueError("replay.learning_starts must be >= replay.batch_size")

    def obs_values(td):
        return [td.get(key) for key in obs_keys]

    checkpoints: list[str] = []

    def save(name: str, frames: int, env_steps: int):
        path = checkpoint_dir / f"{name}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(
            {
                "actor": actor.state_dict(),
                "q1": q1.state_dict(),
                "q2": q2.state_dict(),
                "target_q1": target_q1.state_dict(),
                "target_q2": target_q2.state_dict(),
                "actor_optim": actor_optim.state_dict(),
                "q_optim": q_optim.state_dict(),
                "alpha_optim": alpha_optim.state_dict(),
                "log_alpha": log_alpha.detach().cpu(),
                "frames": frames,
                "env_steps": env_steps,
                "config": config_plain,
            },
            tmp,
        )
        tmp.replace(path)
        write_json(
            path.with_suffix(".json"),
            {
                "checkpoint": path.name,
                "frames": frames,
                "env_steps": env_steps,
                "sha256": sha256_file(path),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "strict_eval": None,
                "config": config_plain,
            },
        )
        checkpoints.append(path.name)
        update_json(manifest_path, frames=frames, checkpoints=checkpoints)
        print("CHECKPOINT " + json.dumps({"path": str(path), "frames": frames}), flush=True)
        return path

    def update_once():
        obs, action, reward, next_obs, terminated = replay.sample(
            int(cfg.replay.batch_size), device
        )
        with torch.no_grad():
            next_action, next_log_prob = actor.sample(*next_obs)
            target_q = torch.minimum(
                target_q1(*next_obs, next_action), target_q2(*next_obs, next_action)
            ) - log_alpha.exp() * next_log_prob
            backup = float(cfg.reward_scale) * reward + float(cfg.sac.gamma) * (1.0 - terminated) * target_q
        q1_value, q2_value = q1(*obs, action), q2(*obs, action)
        q_loss = (q1_value - backup).square().mean() + (q2_value - backup).square().mean()
        q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), float(cfg.optim.max_grad_norm))
        q_optim.step()

        for parameter in list(q1.parameters()) + list(q2.parameters()):
            parameter.requires_grad_(False)
        sampled_action, log_prob = actor.sample(*obs)
        actor_loss = (log_alpha.exp().detach() * log_prob - torch.minimum(
            q1(*obs, sampled_action), q2(*obs, sampled_action)
        )).mean()
        actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), float(cfg.optim.max_grad_norm))
        actor_optim.step()
        for parameter in list(q1.parameters()) + list(q2.parameters()):
            parameter.requires_grad_(True)

        alpha_loss = -(log_alpha * (log_prob.detach() + target_entropy)).mean()
        alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_optim.step()
        su.soft_update(q1, target_q1, float(cfg.sac.tau))
        su.soft_update(q2, target_q2, float(cfg.sac.tau))
        return float(q_loss.detach()), float(actor_loss.detach()), float(alpha_loss.detach())

    stop = {"reason": "total_frames"}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(reason="sigterm"))
    td = env.reset()
    frames = env_steps = 0
    next_checkpoint = int(cfg.checkpoint.interval_frames)
    next_log = int(cfg.logger.interval_frames)
    start = time.monotonic()
    outcome_counts = {name: 0 for name in ("success", "cube_exited_zone", "cube_off_belt", "time_out")}
    episodes = 0
    loss_sums = [0.0, 0.0, 0.0]
    updates = 0
    print(
        "RUN_INFO " + json.dumps(
            {
                "run_dir": str(run_dir), "num_envs": num_envs, "total_frames": total_frames,
                "action_dim": action_dim, "obs": {"/".join(k): list(s) for k, s in zip(obs_keys, shapes)},
                "replay_capacity": replay.capacity,
                "max_episode_length": int(env.base_env._env.unwrapped.max_episode_length),
            }
        ), flush=True,
    )
    try:
        while frames < total_frames and stop["reason"] == "total_frames":
            if cfg.max_hours and time.monotonic() - start >= float(cfg.max_hours) * 3600:
                stop["reason"] = "max_hours"
                break
            with torch.no_grad():
                policy_action, _ = actor.sample(*obs_values(td))
            policy_action = policy_action.clone()
            td.set("action", policy_action)
            transition, continuation = env.step_and_maybe_reset(td)
            next_reward = transition.get(("next", "reward"))
            terminated = transition.get(("next", "terminated"))
            replay.add(
                obs_values(td), policy_action, next_reward, obs_values(continuation), terminated
            )
            done = transition.get(("next", "done")).reshape(-1)
            episodes += int(done.sum())
            for name in outcome_counts:
                flags = transition.get(("next", "outcome", name)).reshape(-1)
                outcome_counts[name] += int((flags & done).sum())
            td = continuation
            env_steps += 1
            frames += num_envs

            if replay.size >= int(cfg.replay.learning_starts):
                for _ in range(int(cfg.sac.gradient_steps)):
                    losses = update_once()
                    loss_sums = [a + b for a, b in zip(loss_sums, losses)]
                    updates += 1

            if frames >= next_log:
                elapsed = max(time.monotonic() - start, 1.0e-6)
                metrics = {
                    "frames": frames,
                    "env_steps": env_steps,
                    "replay_size": replay.size,
                    "updates": updates,
                    "alpha": float(log_alpha.exp().detach()),
                    "frames_per_hour": frames / elapsed * 3600.0,
                    "episodes": episodes,
                    "success_rate": outcome_counts["success"] / episodes if episodes else 0.0,
                    **{f"outcome/{name}": count / episodes if episodes else 0.0 for name, count in outcome_counts.items()},
                }
                if updates:
                    metrics.update(
                        q_loss=loss_sums[0] / updates,
                        actor_loss=loss_sums[1] / updates,
                        alpha_loss=loss_sums[2] / updates,
                    )
                print("METRICS " + json.dumps(metrics), flush=True)
                outcome_counts = {name: 0 for name in outcome_counts}
                episodes = 0
                loss_sums = [0.0, 0.0, 0.0]
                updates = 0
                while next_log <= frames:
                    next_log += int(cfg.logger.interval_frames)

            if frames >= next_checkpoint:
                save(f"sac_sphere_gap_{frames}", frames, env_steps)
                while next_checkpoint <= frames:
                    next_checkpoint += int(cfg.checkpoint.interval_frames)
    except BaseException as error:
        stop["reason"] = f"error: {type(error).__name__}"
        raise
    finally:
        save("sac_sphere_gap_final", frames, env_steps)
        update_json(
            manifest_path,
            stop_reason=stop["reason"], frames=frames,
            ended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        print("SAC_DONE " + json.dumps({"reason": stop["reason"], "frames": frames}), flush=True)
        env.close()


if __name__ == "__main__":
    main()
