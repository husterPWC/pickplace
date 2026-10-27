#!/usr/bin/env python3
"""Dump a deterministic Isaac reset observation/action for MuJoCo contract comparison."""

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/teachers/teacher_repro_v3c_4096_no_rtx/checkpoints/ppo_teacher_30015488.pt"

from pickplace.app import launch_app

app = launch_app(headless=True, enable_cameras=False)

import torch  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402
from pickplace.torchrl_env import make_env  # noqa: E402

spec = importlib.util.spec_from_file_location("teacher_utils_contract", ROOT / "pipeline/0_state_teacher/utils.py")
tu = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tu
spec.loader.exec_module(tu)

payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
cfg = dict(payload["config"]["env"])
cfg.update(
    num_envs=1,
    cameras=False,
    seed=0,
    robot_reset={"position_range": [0.0, 0.0], "velocity_range": [0.0, 0.0]},
    belt={
        **cfg["belt"],
        "pallet_start_range": [0.0, 0.0],
        "bowl_offset_x": [0.0, 0.0],
        "bowl_offset_y": [0.0, 0.0],
    },
    food_params={"spawn_range": 0.0},
)
env = make_env(cfg)
actor = tu.load_teacher_actor(CHECKPOINT, env, env.device)
td = env.reset()
keys = [
    ("proprio", "ee_pos"), ("proprio", "ee_quat"), ("proprio", "gripper_pos"),
    ("proprio", "joint_pos_rel"), ("proprio", "joint_vel_rel"), ("proprio", "last_action"),
    ("belt", "bowl_pos"), ("privileged", "food_pos"), ("privileged", "food_quat"),
    ("privileged", "is_grasped"),
]
with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    actor(td)
record = {"keys": [list(k) for k in keys]}
record["values"] = {"/".join(k): td[k][0].detach().cpu().tolist() for k in keys}
record["flat"] = torch.cat([td[k][0].reshape(-1) for k in keys]).detach().cpu().tolist()
record["action"] = td["action"][0].detach().cpu().tolist()
record["actor_in_keys"] = [list(k) if isinstance(k, tuple) else k for k in actor.in_keys]
record["rollout"] = []
with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    for step in range(400):
        td = actor(td)
        action = td["action"][0].detach().cpu().tolist()
        stepped, td = env.step_and_maybe_reset(td)
        done = bool(stepped["next", "done"][0])
        outcome = [
            name for name in ("success", "bowl_exited_zone", "bowl_off_belt", "bowl_tipped", "food_off_table", "time_out")
            if bool(stepped["next", "outcome", name][0])
        ]
        record["rollout"].append(
            {
                "step": step + 1,
                "action": action,
                "ee_pos": td["proprio", "ee_pos"][0].detach().cpu().tolist(),
                "joint_pos_rel": td["proprio", "joint_pos_rel"][0].detach().cpu().tolist(),
                "joint_vel_rel": td["proprio", "joint_vel_rel"][0].detach().cpu().tolist(),
                "gripper_pos": td["proprio", "gripper_pos"][0].detach().cpu().tolist(),
                "food_pos": td["privileged", "food_pos"][0].detach().cpu().tolist(),
                "bowl_pos": td["belt", "bowl_pos"][0].detach().cpu().tolist(),
                "is_grasped": td["privileged", "is_grasped"][0].detach().cpu().tolist(),
                "done": done,
                "outcome": outcome,
            }
        )
        if done:
            break
out = ROOT / "artifacts/mujoco/isaac_fixed_contract.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=2) + "\n")
print("ISAAC_CONTRACT " + json.dumps({"steps": len(record["rollout"]), "last": record["rollout"][-1]}), flush=True)
print(f"ISAAC_CONTRACT_DONE {out}", flush=True)
os._exit(0)
