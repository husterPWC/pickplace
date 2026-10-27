"""Distil C0m behaviour into the strictly speed-free C0p actor.

The rollout environment temporarily exposes both observation contracts.  Only
the frozen C0m teacher reads the legacy velocity-bearing leaves.  The student
reads the keys declared in C0p and the saved checkpoint stores C0p's pure
speed-free environment config.
"""

import json
import os
import sys
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_utils():
    import importlib.util

    spec = importlib.util.spec_from_file_location("teacher_distill_utils", os.path.join(HERE, "utils.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["teacher_distill_utils"] = module
    spec.loader.exec_module(module)
    return module


@hydra.main(config_path="", config_name="config_moving_cube_c0p_gap_speed_free", version_base="1.3")
def main(cfg: DictConfig):
    from pickplace.app import launch_app

    if cfg.env.cameras:
        raise ValueError("C0p state distillation requires env.cameras=false")
    launch_app(headless=cfg.app.headless, enable_cameras=False, device=cfg.env.device)

    import torch
    from torchrl.collectors import Collector

    from pickplace.artifacts import git_commit, new_run_dir, sha256_file, write_json
    from pickplace.keys import expand_in_keys
    from pickplace.torchrl_env import make_env

    tu = _load_utils()
    teacher_path = Path(cfg.distill.teacher_checkpoint).resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {teacher_path}")

    # This expanded spec exists only during collection.  It is never written
    # into the C0p checkpoint and therefore cannot leak into deployment.
    rollout_env_cfg = OmegaConf.to_container(cfg.env, resolve=True)
    rollout_env_cfg.update(
        privileged_information=True,
        grasp_observation=False,
        kinematic_grasp_proxy_observation=True,
        temporal_history_steps=10,
        speed_free_observations=False,
    )
    device = torch.device(cfg.env.device)
    torch.manual_seed(int(cfg.env.seed))
    env = make_env(rollout_env_cfg)
    teacher = tu.load_teacher_actor(teacher_path, env, device).eval().requires_grad_(False)
    student, critic = tu.make_teacher_models(env, cfg.network, device)

    rollout_policy = str(cfg.distill.rollout_policy)
    if rollout_policy not in ("teacher", "student"):
        raise ValueError("distill.rollout_policy must be 'teacher' or 'student'")
    init_student = cfg.distill.init_student_checkpoint
    if init_student:
        init_path = Path(init_student).resolve()
        state = torch.load(init_path, map_location=device, weights_only=False)
        student.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
    elif rollout_policy == "student":
        raise ValueError("student rollouts require distill.init_student_checkpoint")

    # Match C0m exploration scale for later PPO compatibility; supervised loss
    # changes the deterministic mean only.
    teacher_scales = [p for n, p in teacher.named_parameters() if n.endswith("state_independent_scale")]
    student_scales = [p for n, p in student.named_parameters() if n.endswith("state_independent_scale")]
    if len(teacher_scales) != 1 or len(student_scales) != 1:
        raise RuntimeError("expected exactly one state-independent actor scale")
    with torch.no_grad():
        if not init_student:
            student_scales[0].copy_(teacher_scales[0])
    student_scales[0].requires_grad_(False)

    forbidden = {"joint_vel_rel", "food_lin_vel", "speed", "motion", "is_grasped", "grasp_proxy"}
    actor_keys = expand_in_keys(env.observation_spec, list(cfg.network.actor_in_keys))
    bad = [key for key in actor_keys if any(part in forbidden for part in key)]
    if bad:
        raise RuntimeError(f"C0p speed-free contract violated by actor keys: {bad}")

    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optim = torch.optim.Adam(parameters, lr=float(cfg.optim.lr), eps=1e-5)
    rollout_steps = int(cfg.collector.rollout_steps)
    frames_per_batch = int(cfg.env.num_envs) * rollout_steps
    total_frames = int(cfg.collector.total_frames)
    iterations = max(1, (total_frames + frames_per_batch - 1) // frames_per_batch)
    mini_batch = int(cfg.distill.mini_batch_size)
    epochs = int(cfg.distill.epochs_per_rollout)

    run_dir = Path(cfg.run.dir) if cfg.run.dir else new_run_dir("teachers", cfg.run.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "speed_free_policy_distillation",
        "git_commit": git_commit(),
        "teacher_checkpoint": str(teacher_path),
        "teacher_sha256": sha256_file(teacher_path),
        "rollout_policy": rollout_policy,
        "init_student_checkpoint": str(Path(init_student).resolve()) if init_student else None,
        "speed_free_actor_keys": [list(key) for key in actor_keys],
        "config": OmegaConf.to_container(cfg, resolve=True),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ended_at": None,
    }
    write_json(run_dir / "manifest.json", manifest)
    print("RUN_INFO " + json.dumps({"run_dir": str(run_dir), "frames_per_batch": frames_per_batch,
                                     "iterations": iterations, "actor_keys": manifest["speed_free_actor_keys"]}), flush=True)

    class DeterministicActor(torch.nn.Module):
        """Collector policy that executes tanh(mean) independent of global exploration context."""

        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, tensordict):
            distribution = self.actor.get_dist(tensordict, tensordict_out=tensordict)
            tensordict.set("action", distribution.loc.tanh())
            return tensordict

    collector_policy = teacher if rollout_policy == "teacher" else DeterministicActor(student)
    collector = Collector(
        env, collector_policy, frames_per_batch=frames_per_batch, total_frames=-1,
        device=device, no_cuda_sync=True, trust_policy=True,
    )
    frames = 0
    next_checkpoint = int(cfg.distill.checkpoint_interval_frames)
    data_iter = iter(collector)
    try:
        for iteration in range(iterations):
            data = next(data_iter)
            flat = data.reshape(-1)
            rollout_noise_mae = (flat.get("action") - flat.get("loc").tanh()).abs().mean().item()
            if rollout_policy == "student" and rollout_noise_mae > 1e-7:
                raise RuntimeError(
                    f"deterministic student rollout contract violated: action MAE={rollout_noise_mae}"
                )
            with torch.no_grad():
                target_loc = teacher.get_dist(flat, tensordict_out=flat.empty()).loc.detach().clone()
            count = flat.numel()
            loss_sum, updates = 0.0, 0
            student.train()
            for _ in range(epochs):
                for indices in torch.randperm(count, device=device).split(mini_batch):
                    batch = flat[indices]
                    predicted = student.get_dist(batch, tensordict_out=batch.empty()).loc
                    loss = torch.nn.functional.smooth_l1_loss(predicted, target_loc[indices])
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optim.step()
                    loss_sum += float(loss.detach())
                    updates += 1
            frames += count
            mean_loss = loss_sum / max(updates, 1)
            with torch.no_grad():
                predicted_loc = student.get_dist(flat, tensordict_out=flat.empty()).loc
                action_error = (predicted_loc.tanh() - target_loc.tanh()).abs()
                action_mae = action_error.mean().item()
                action_mae_by_dim = action_error.mean(dim=0).tolist()
                gripper_sign_agreement = (
                    predicted_loc[:, -1].sign() == target_loc[:, -1].sign()
                ).float().mean().item()
            print("METRICS " + json.dumps({"iteration": iteration, "frames": frames,
                                             "distill/loss": mean_loss,
                                             "distill/action_mae": action_mae,
                                             "distill/action_mae_by_dim": action_mae_by_dim,
                                             "distill/gripper_sign_agreement": gripper_sign_agreement,
                                             "rollout/noise_mae": rollout_noise_mae}), flush=True)
            if rollout_policy == "student":
                collector.update_policy_weights_()
            if next_checkpoint > 0 and frames >= next_checkpoint:
                path = tu.save_teacher_checkpoint(
                    run_dir, f"ppo_teacher_{frames}", student, critic, optim, cfg, frames, iteration
                )
                print("CHECKPOINT " + json.dumps({"path": str(path), "frames": frames}), flush=True)
                while next_checkpoint <= frames:
                    next_checkpoint += int(cfg.distill.checkpoint_interval_frames)
    finally:
        final = tu.save_teacher_checkpoint(
            run_dir, "ppo_teacher_final", student, critic, optim, cfg, frames, max(iterations - 1, 0)
        )
        collector.shutdown()
        manifest.update(ended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), frames=frames,
                        final_checkpoint=str(final))
        write_json(run_dir / "manifest.json", manifest)
        print("DISTILL_DONE " + json.dumps({"frames": frames, "checkpoint": str(final)}), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
