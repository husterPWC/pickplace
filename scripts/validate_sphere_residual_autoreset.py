#!/usr/bin/env python3
"""Verify two strict successes per env across Isaac native auto-reset."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
cli = OmegaConf.from_dotlist(sys.argv[1:])
checkpoint = Path(cli.checkpoint).resolve()
num_envs = int(cli.get("num_envs", 4))
episodes_per_env = int(cli.get("episodes_per_env", 2))
speed = float(cli.get("speed", 0.03))
zone_length = float(cli.get("zone_length", 0.40))
seed = int(cli.get("seed", 0))

from pickplace.app import launch_app  # noqa: E402

launch_app(headless=True, enable_cameras=False)

import torch  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from pickplace.artifacts import write_json  # noqa: E402
from pickplace.torchrl_env import make_env  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "teacher_utils_autoreset", ROOT / "pipeline/0_state_teacher/utils.py"
)
teacher_utils = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = teacher_utils
spec.loader.exec_module(teacher_utils)

payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
env_cfg = OmegaConf.to_container(
    OmegaConf.merge(
        payload["config"]["env"],
        {
            "num_envs": num_envs,
            "cameras": False,
            "seed": seed,
            # Belt.place_window is a duration.  Keep the physical interception
            # zone fixed while selecting a deterministic speed bucket.
            "belt": {
                "speed": speed,
                "speed_noise": 0.0,
                "place_window": zone_length / speed,
            },
        },
    ),
    resolve=True,
)
torch.manual_seed(seed)
env = make_env(env_cfg)
actor = teacher_utils.load_teacher_actor(checkpoint, env, env.device)
td = env.reset()
counts = torch.zeros(num_envs, dtype=torch.long, device=env.device)
successes = torch.zeros(num_envs, dtype=torch.long, device=env.device)
outcomes: list[list[str]] = [[] for _ in range(num_envs)]
max_steps = (int(env.base_env._env.unwrapped.max_episode_length) + 1) * episodes_per_env

with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    for step in range(max_steps):
        td = actor(td)
        transition, td = env.step_and_maybe_reset(td)
        done = transition["next", "done"].reshape(num_envs).bool()
        success = transition["next", "outcome", "success"].reshape(num_envs).bool()
        for index in done.nonzero(as_tuple=False).flatten().tolist():
            if counts[index] >= episodes_per_env:
                continue
            fired = [
                name
                for name, value in transition["next", "outcome"].items()
                if bool(value.reshape(num_envs)[index])
            ]
            outcomes[index].append(",".join(fired) or "unknown")
            counts[index] += 1
            successes[index] += int(success[index])
        if bool((counts >= episodes_per_env).all()):
            break

record = {
    "checkpoint": str(checkpoint),
    "speed": speed,
    "zone_length": zone_length,
    "seed": seed,
    "num_envs": num_envs,
    "episodes_per_env": episodes_per_env,
    "steps": step + 1,
    "counts": counts.cpu().tolist(),
    "successes": successes.cpu().tolist(),
    "outcomes": outcomes,
    "passed": bool((successes == episodes_per_env).all()),
}
output = checkpoint.parent.parent / f"autoreset_speed_{speed:.3f}.json"
write_json(output, record)
print("SPHERE_RESIDUAL_AUTORESET " + json.dumps(record), flush=True)
env.close()
if not record["passed"]:
    raise SystemExit(1)
os._exit(0)
