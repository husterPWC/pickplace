#!/usr/bin/env python3
"""Run the Isaac-trained v3c state teacher closed-loop in MuJoCo.

Examples:
    .venv/bin/python -m mujoco_sim.policy_runner episodes=20
    .venv/bin/python -m mujoco_sim.policy_runner viewer=true episodes=3 realtime=true
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mujoco_sim.env import FrankaPickPlaceMuJoCo, POLICY_DT, ROOT
from mujoco_sim.expert import MuJoCoPickPlaceExpert


DEFAULT_CHECKPOINT = (
    ROOT
    / "artifacts/teachers/teacher_repro_v3c_4096_no_rtx/checkpoints/ppo_teacher_30015488.pt"
)


def parse_cli() -> dict[str, str]:
    return dict(arg.split("=", 1) for arg in sys.argv[1:] if "=" in arg)


def as_bool(value: str | bool) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


class TeacherActor(nn.Module):
    """Minimal deterministic reconstruction of the checkpoint's TorchRL TanhNormal actor."""

    def __init__(self, checkpoint: Path):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = payload["config"]
        if config["network"]["actor_in_keys"] != ["proprio", "belt", "privileged"]:
            raise ValueError("checkpoint observation groups do not match the MuJoCo state contract")
        hidden = list(config["network"]["hidden"])
        if hidden != [512, 256, 128]:
            raise ValueError(f"unexpected teacher hidden sizes: {hidden}")
        self.layers = nn.ModuleList(
            [nn.Linear(45, 512), nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 7)]
        )
        state = payload["actor"]
        for index, layer in zip((0, 2, 4, 6), self.layers):
            layer.weight.data.copy_(state[f"module.0.module.body.mlp.{index}.weight"])
            layer.bias.data.copy_(state[f"module.0.module.body.mlp.{index}.bias"])
        self.eval()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        value = observation
        for layer in self.layers[:-1]:
            value = torch.nn.functional.elu(layer(value))
        # TorchRL's deterministic_sample for TanhNormal is the transformed root mean.
        return torch.tanh(self.layers[-1](value))

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        return self(torch.from_numpy(observation).float().unsqueeze(0))[0].cpu().numpy()

    def reset(self) -> None:
        pass


class PhaseConditionedActor(nn.Module):
    """v3c-warm-started actor with a minimal high-level task-phase memory."""

    def __init__(self, checkpoint: Path):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        spec = payload["sim2sim_phase_actor"]
        self.phases = tuple(spec["phases"])
        self.nominal_blend = float(spec.get("nominal_blend", 0.95))
        self.layers = nn.ModuleList(
            [nn.Linear(45 + len(self.phases), 512), nn.Linear(512, 256), nn.Linear(256, 128), nn.Linear(128, 7)]
        )
        self.load_state_dict(spec["state_dict"])
        self.tracker = MuJoCoPickPlaceExpert()
        self.eval()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        value = observation
        for layer in self.layers[:-1]:
            value = torch.nn.functional.elu(layer(value))
        return torch.tanh(self.layers[-1](value))

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        phase = self.tracker.phase
        # The dynamics-identified controller is the safe nominal action; the
        # learned phase-conditioned actor contributes the residual fraction.
        nominal_action = self.tracker.act(observation)
        phase_one_hot = np.zeros(len(self.phases), dtype=np.float32)
        phase_one_hot[self.phases.index(phase)] = 1.0
        augmented = np.concatenate([observation, phase_one_hot])
        learned_action = self(torch.from_numpy(augmented).float().unsqueeze(0))[0].cpu().numpy()
        return np.clip(
            self.nominal_blend * nominal_action + (1.0 - self.nominal_blend) * learned_action,
            -1.0,
            1.0,
        )

    def reset(self) -> None:
        self.tracker.reset()


def load_actor(checkpoint: Path) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "sim2sim_phase_actor" in payload:
        return PhaseConditionedActor(checkpoint)
    return TeacherActor(checkpoint)


def run() -> None:
    cli = parse_cli()
    checkpoint = Path(cli.get("checkpoint", DEFAULT_CHECKPOINT)).resolve()
    episodes = int(cli.get("episodes", 20))
    seed = int(cli.get("seed", 0))
    viewer_enabled = as_bool(cli.get("viewer", False))
    realtime = as_bool(cli.get("realtime", viewer_enabled))
    viewer_hold = float(cli.get("viewer_hold", 0.0))
    randomize = as_bool(cli.get("randomize", True))
    grasp_assist = as_bool(cli.get("grasp_assist", True))
    belt_speed = float(cli.get("belt_speed", 0.08))
    out = Path(cli.get("out", ROOT / "artifacts/mujoco/franka_v3c_sim2sim.json"))
    actor = load_actor(checkpoint)
    env = FrankaPickPlaceMuJoCo(
        seed=seed,
        randomize=randomize,
        grasp_assist=grasp_assist,
        belt_speed=belt_speed,
    )
    viewer = None
    if viewer_enabled:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.cam.lookat[:] = (0.38, 0.05, 0.28)
        viewer.cam.distance = 1.65
        viewer.cam.azimuth = 132
        viewer.cam.elevation = -25

    records = []
    try:
        for episode in range(episodes):
            actor.reset()
            obs = env.reset(seed + episode)
            step = 0
            peak_grasp = False
            min_home_after_grasp = float("inf")
            while True:
                wall_start = time.perf_counter()
                action = actor.act(obs)
                result = env.step(action)
                obs = result.observation
                step += 1
                peak_grasp |= bool(result.info["is_grasped"])
                if peak_grasp:
                    min_home_after_grasp = min(min_home_after_grasp, float(result.info["home_distance"]))
                if viewer is not None:
                    if not viewer.is_running():
                        return
                    viewer.sync()
                if realtime:
                    remaining = POLICY_DT - (time.perf_counter() - wall_start)
                    if remaining > 0:
                        time.sleep(remaining)
                if result.done:
                    record = {
                        "episode": episode,
                        "seed": seed + episode,
                        "outcome": result.outcome,
                        "steps": step,
                        "seconds": round(step * POLICY_DT, 3),
                        "ever_grasped": peak_grasp,
                        "min_home_after_grasp": None if not peak_grasp else round(min_home_after_grasp, 5),
                        **result.info,
                    }
                    records.append(record)
                    print("MUJOCO_EPISODE " + json.dumps(record), flush=True)
                    if viewer is not None and viewer_hold > 0:
                        deadline = time.monotonic() + viewer_hold
                        while viewer.is_running() and time.monotonic() < deadline:
                            viewer.sync()
                            time.sleep(0.02)
                    break
    finally:
        if viewer is not None:
            viewer.close()
        env.close()

    counts = Counter(r["outcome"] for r in records)
    summary = {
        "checkpoint": str(checkpoint),
        "engine": "mujoco",
        "mujoco_version": __import__("mujoco").__version__,
        "episodes": len(records),
        "successes": counts.get("success", 0),
        "success_rate": counts.get("success", 0) / max(1, len(records)),
        "outcomes": dict(counts),
        "settings": {
            "seed": seed,
            "randomize": randomize,
            "grasp_assist": grasp_assist,
            "belt_speed": belt_speed,
            "policy_dt": POLICY_DT,
        },
        "records": records,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print("MUJOCO_SIM2SIM " + json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)
    print(f"MUJOCO_SIM2SIM_DONE {out}", flush=True)

    # GLFW can segfault during Python interpreter teardown on some Linux/NVIDIA
    # combinations even after a clean passive-viewer close.  All MuJoCo and
    # output resources are already closed/written here, so bypass that teardown
    # path for viewer runs.  Headless runs and imported tests return normally.
    if viewer_enabled:
        os._exit(0)


if __name__ == "__main__":
    run()
