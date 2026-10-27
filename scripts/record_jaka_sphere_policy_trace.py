#!/usr/bin/env python3
"""Record one deterministic JAKA sphere-policy rollout without RTX cameras."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

from pickplace.app import launch_app


def _options(argv: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in argv:
        if "=" not in item:
            raise SystemExit("usage: record_jaka_sphere_policy_trace.py checkpoint=... [out=...] [seed=31]")
        key, value = item.split("=", 1)
        result[key] = value
    return result


opts = _options(sys.argv[1:])
if "checkpoint" not in opts:
    raise SystemExit("checkpoint=<ppo_teacher.pt> is required")
checkpoint = Path(opts["checkpoint"]).resolve()
output = Path(
    opts.get(
        "out",
        str(checkpoint.parent.parent / "videos" / f"{checkpoint.stem}_deterministic_trace.npz"),
    )
).resolve()
seed = int(opts.get("seed", 31))
stride = int(opts.get("stride", 2))
max_steps = int(opts.get("max_steps", 1200))
if stride < 1 or max_steps < 1:
    raise SystemExit("stride and max_steps must be positive")

# Device selection is deliberately supplied by the launcher environment.  On
# this workstation PICKPLACE_PHYSICAL_GPU=1 maps physical GPU1 to cuda:0.
app = launch_app(headless=True, enable_cameras=False, device="cuda:0")

import torch  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from pickplace.torchrl_env import make_env  # noqa: E402


def _teacher_utils():
    path = Path(__file__).resolve().parents[1] / "pipeline/0_state_teacher/utils.py"
    spec = importlib.util.spec_from_file_location("jaka_video_teacher_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tensor(value):
    return value.torch if hasattr(value, "torch") else value


state = torch.load(checkpoint, map_location="cpu", weights_only=False)
env_cfg = dict(state["config"]["env"])
if env_cfg.get("arm") != "zu5_ag95":
    raise RuntimeError(f"refusing to label a non-JAKA checkpoint as JAKA: arm={env_cfg.get('arm')!r}")
env_cfg.update(num_envs=1, cameras=False, render_camera=False, seed=seed)
env = make_env(env_cfg)
actor = _teacher_utils().load_teacher_actor(checkpoint, env, env.device).eval()
u = env.base_env._env.unwrapped
skill = next(
    transform.controller
    for transform in env.transform.transforms
    if getattr(transform, "controller", None).__class__.__name__
    in {"StableSphereResidualSkill", "StableSpherePickPlaceSkill"}
)
residual = skill.grasp if skill.__class__.__name__ == "StableSpherePickPlaceSkill" else skill

robot = u.scene["robot"]
pallet = u.scene["pallet"]
food = u.scene["food"]
bowl = u.scene["bowl"]
dt = float(u.step_dt)
origin = u.scene.env_origins[0]
trace: dict[str, list] = {
    "time": [],
    "body_pos": [],
    "body_quat": [],
    "pallet_pos": [],
    "pallet_quat": [],
    "sphere_pos": [],
    "sphere_quat": [],
    "bowl_pos": [],
    "bowl_quat": [],
    "tcp_pos": [],
    "pose_rotation_error": [],
    "phase": [],
    "transport_phase": [],
    "gripper_pos": [],
}


def record(step: int, td) -> None:
    phase_id = 0 if residual.phase is None else int(residual.phase[0])
    trace["time"].append(step * dt)
    trace["body_pos"].append((_tensor(robot.data.body_link_pos_w)[0] - origin).detach().cpu().numpy())
    trace["body_quat"].append(_tensor(robot.data.body_link_quat_w)[0].detach().cpu().numpy())
    trace["pallet_pos"].append((_tensor(pallet.data.body_link_pos_w)[0] - origin).detach().cpu().numpy())
    trace["pallet_quat"].append(_tensor(pallet.data.body_link_quat_w)[0].detach().cpu().numpy())
    trace["sphere_pos"].append((_tensor(food.data.root_pos_w)[0] - origin).detach().cpu().numpy())
    trace["sphere_quat"].append(_tensor(food.data.root_quat_w)[0].detach().cpu().numpy())
    trace["bowl_pos"].append((_tensor(bowl.data.root_pos_w)[0] - origin).detach().cpu().numpy())
    trace["bowl_quat"].append(_tensor(bowl.data.root_quat_w)[0].detach().cpu().numpy())
    trace["tcp_pos"].append(
        (_tensor(u.scene["ee_frame"].data.target_pos_w)[0, 0] - origin).detach().cpu().numpy()
    )
    rotation = td["grasp_guidance", "best_pose_delta"][0, 3:]
    trace["pose_rotation_error"].append(float(torch.linalg.vector_norm(rotation)))
    trace["phase"].append(residual.baseline.PHASE_NAMES[phase_id])
    if skill.__class__.__name__ == "StableSpherePickPlaceSkill":
        transport_phase_id = 0 if skill.phase is None else int(skill.phase[0])
        trace["transport_phase"].append(skill.PHASE_NAMES[transport_phase_id])
    else:
        trace["transport_phase"].append("not_enabled")
    trace["gripper_pos"].append(float(td["proprio", "gripper_pos"][0, 0]))


td = env.reset()
record(0, td)
outcome = "unfinished"
steps = 0
with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    for step in range(1, max_steps + 1):
        td = actor(td)
        transition, td = env.step_and_maybe_reset(td)
        steps = step
        done = bool(transition["next", "done"][0])
        if step % stride == 0 and not done:
            record(step, td)
        if done:
            names = (
                "success", "bowl_exited_zone", "bowl_off_belt", "bowl_tipped",
                "food_off_table", "cube_exited_zone", "cube_off_belt", "tcp_escape", "time_out",
            )
            outcome = next(
                (name for name in names if bool(transition["next", "outcome", name][0])),
                "unknown",
            )
            break

output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    output,
    **{key: np.asarray(value) for key, value in trace.items()},
    body_names=np.asarray(robot.body_names),
    pallet_body_names=np.asarray(pallet.body_names),
    dt=np.asarray(dt * stride),
    outcome=np.asarray(outcome),
    checkpoint=np.asarray(str(checkpoint)),
    seed=np.asarray(seed),
    steps=np.asarray(steps),
    belt_center=np.asarray((u.cfg.belt.zone_center_x, u.cfg.belt.belt_y)),
    belt_half_width=np.asarray(u.cfg.belt.belt_half_width),
    belt_visual_length=np.asarray(u.cfg.belt.visual_length()),
)
report = {
    "trace": str(output),
    "checkpoint": str(checkpoint),
    "seed": seed,
    "steps": steps,
    "frames": len(trace["time"]),
    "outcome": outcome,
    "initial_rotation_error_rad": trace["pose_rotation_error"][0],
    "max_rotation_error_rad": max(trace["pose_rotation_error"]),
    "physical_gpu": os.environ.get("PICKPLACE_PHYSICAL_GPU"),
}
output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("JAKA_POLICY_TRACE " + json.dumps(report), flush=True)
env.close()
os._exit(0)
