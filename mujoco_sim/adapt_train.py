#!/usr/bin/env python3
"""Warm-start the Isaac v3c actor from privileged MuJoCo expert rollouts."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from mujoco_sim.env import FrankaPickPlaceMuJoCo, ROOT
from mujoco_sim.expert import MuJoCoPickPlaceExpert
from mujoco_sim.policy_runner import DEFAULT_CHECKPOINT, TeacherActor, as_bool, parse_cli


DEFAULT_DIR = ROOT / "artifacts/mujoco/adaptation"
PHASES = MuJoCoPickPlaceExpert.phases


class PhaseConditionedTrainingActor(torch.nn.Module):
    """Expand the original 45-D v3c actor with zero-initialized phase inputs."""

    def __init__(self, source: Path):
        super().__init__()
        base = TeacherActor(source)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(45 + len(PHASES), 512), torch.nn.Linear(512, 256), torch.nn.Linear(256, 128), torch.nn.Linear(128, 7)]
        )
        with torch.no_grad():
            self.layers[0].weight.zero_()
            self.layers[0].weight[:, :45].copy_(base.layers[0].weight)
            self.layers[0].bias.copy_(base.layers[0].bias)
            for target, original in zip(self.layers[1:], base.layers[1:]):
                target.weight.copy_(original.weight)
                target.bias.copy_(original.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        value = observation
        for layer in self.layers[:-1]:
            value = torch.nn.functional.elu(layer(value))
        return torch.tanh(self.layers[-1](value))


def collect_episode(seed: int) -> dict:
    env = FrankaPickPlaceMuJoCo(seed=seed, randomize=True, grasp_assist=True)
    expert = MuJoCoPickPlaceExpert()
    observation = env.reset(seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    phases: list[str] = []
    outcome = "step_limit"
    for _ in range(600):
        phase = expert.phase
        action = expert.act(observation)
        observations.append(observation.copy())
        actions.append(action.copy())
        phases.append(phase)
        result = env.step(action)
        observation = result.observation
        if result.done:
            outcome = str(result.outcome)
            break
    return {
        "seed": seed,
        "outcome": outcome,
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "phases": np.asarray(phases),
    }


def collect_dataset(episodes: int, seed: int, workers: int, path: Path) -> dict[str, np.ndarray]:
    seeds = list(range(seed, seed + episodes))
    if workers <= 1:
        records = [collect_episode(value) for value in seeds]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(collect_episode, seeds))

    outcomes = Counter(record["outcome"] for record in records)
    print("ADAPT_COLLECT " + json.dumps({"episodes": episodes, "outcomes": outcomes}), flush=True)
    failed = [record["seed"] for record in records if record["outcome"] != "success"]
    if failed:
        raise RuntimeError(f"expert failed collection seeds: {failed}")

    episode_ids = np.concatenate(
        [np.full(len(record["actions"]), index, dtype=np.int32) for index, record in enumerate(records)]
    )
    data = {
        "observations": np.concatenate([record["observations"] for record in records]),
        "actions": np.concatenate([record["actions"] for record in records]),
        "phases": np.concatenate([record["phases"] for record in records]),
        "episode_ids": episode_ids,
        "seeds": np.asarray(seeds, dtype=np.int32),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"ADAPT_DATASET_DONE {path} samples={len(episode_ids)}", flush=True)
    return data


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def train(
    data: dict[str, np.ndarray],
    source: Path,
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    phase_conditioned: bool,
    nominal_blend: float,
) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if not use_cuda:
        torch.set_num_threads(min(16, os.cpu_count() or 1))

    actor = (PhaseConditionedTrainingActor(source) if phase_conditioned else TeacherActor(source)).to(device)
    observations = torch.from_numpy(data["observations"])
    actions = torch.from_numpy(data["actions"])
    phases = data["phases"].astype(str)
    if phase_conditioned:
        phase_indices = np.asarray([PHASES.index(phase) for phase in phases])
        phase_one_hot = torch.nn.functional.one_hot(
            torch.from_numpy(phase_indices), num_classes=len(PHASES)
        ).float()
        observations = torch.cat([observations, phase_one_hot], dim=1)
    counts = Counter(phases.tolist())
    sample_weights = torch.tensor([1.0 / counts[phase] for phase in phases], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = DataLoader(
        TensorDataset(observations, actions),
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=use_cuda,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(actor.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.05)
    action_weights = torch.tensor([1, 1, 1, 0.5, 0.5, 0.5, 2.0], device=device)

    for epoch in range(epochs):
        actor.train()
        total = 0.0
        seen = 0
        for obs_batch, action_batch in loader:
            obs_batch = obs_batch.to(device, non_blocking=True)
            action_batch = action_batch.to(device, non_blocking=True)
            prediction = actor(obs_batch)
            loss = (((prediction - action_batch) ** 2) * action_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(obs_batch)
            seen += len(obs_batch)
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                "ADAPT_EPOCH "
                + json.dumps({"epoch": epoch + 1, "loss": total / seen, "lr": scheduler.get_last_lr()[0]}),
                flush=True,
            )

    actor.cpu().eval()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if phase_conditioned:
        payload["sim2sim_phase_actor"] = {
            "phases": list(PHASES),
            "state_dict": {key: value.detach().cpu() for key, value in actor.state_dict().items()},
            "nominal_blend": nominal_blend,
        }
    else:
        for index, layer in zip((0, 2, 4, 6), actor.layers):
            payload["actor"][f"module.0.module.body.mlp.{index}.weight"] = layer.weight.detach().clone()
            payload["actor"][f"module.0.module.body.mlp.{index}.bias"] = layer.bias.detach().clone()
    payload["sim2sim_adaptation"] = {
        "source_checkpoint": str(source),
        "method": "phase-conditioned privileged behavior cloning" if phase_conditioned else "phase-balanced privileged behavior cloning",
        "samples": int(len(observations)),
        "episodes": int(len(np.unique(data["episode_ids"]))),
        "epochs": epochs,
        "phase_counts": dict(counts),
        "grasp_assist": True,
        "phase_conditioned": phase_conditioned,
        "nominal_blend": nominal_blend if phase_conditioned else 0.0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"ADAPT_CHECKPOINT_DONE {output} device={device}", flush=True)


def main() -> None:
    cli = parse_cli()
    episodes = int(cli.get("episodes", 64))
    seed = int(cli.get("seed", 1000))
    workers = int(cli.get("workers", min(4, os.cpu_count() or 1)))
    epochs = int(cli.get("epochs", 80))
    batch_size = int(cli.get("batch_size", 512))
    learning_rate = float(cli.get("lr", 3e-4))
    phase_conditioned = as_bool(cli.get("phase_conditioned", True))
    nominal_blend = float(cli.get("nominal_blend", 0.95))
    source = Path(cli.get("source", DEFAULT_CHECKPOINT)).resolve()
    dataset = Path(cli.get("dataset", DEFAULT_DIR / "expert_demos.npz")).resolve()
    output = Path(cli.get("output", DEFAULT_DIR / "franka_v3c_mujoco_adapted.pt")).resolve()
    if as_bool(cli.get("stamp_only", False)):
        payload = torch.load(output, map_location="cpu", weights_only=False)
        payload["sim2sim_phase_actor"]["nominal_blend"] = nominal_blend
        payload["sim2sim_adaptation"]["nominal_blend"] = nominal_blend
        temporary = output.with_suffix(output.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, output)
        print(f"ADAPT_CHECKPOINT_STAMPED {output} nominal_blend={nominal_blend}", flush=True)
        return
    recollect = as_bool(cli.get("recollect", not dataset.is_file()))
    data = collect_dataset(episodes, seed, workers, dataset) if recollect else load_dataset(dataset)
    train(data, source, output, epochs, batch_size, learning_rate, phase_conditioned, nominal_blend)


if __name__ == "__main__":
    main()
