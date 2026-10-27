"""Trace one deterministic ZU5+AG95 residual-policy episode.

The residual task's ``tcp_escape`` outcome combines a true Cartesian workspace
escape with sustained retreat from the intercept objective.  This diagnostic
records the TCP and raw actor mean so the two causes cannot be confused.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

from pickplace.app import launch_app


if len(sys.argv) != 2:
    raise SystemExit("usage: diagnose_jaka_residual_policy.py CHECKPOINT.pt")
checkpoint = Path(sys.argv[1]).resolve()
app = launch_app(headless=True, enable_cameras=False, device="cuda:0")

import torch  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from pickplace.torchrl_env import make_env  # noqa: E402


def _teacher_utils():
    path = Path(__file__).resolve().parents[1] / "pipeline/0_state_teacher/utils.py"
    spec = importlib.util.spec_from_file_location("jaka_residual_diag_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


state = torch.load(checkpoint, map_location="cpu", weights_only=False)
env_cfg = dict(state["config"]["env"])
env_cfg.update(num_envs=1, cameras=False, seed=31)
env = make_env(env_cfg)
actor = _teacher_utils().load_teacher_actor(checkpoint, env, env.device).eval()
residual = next(
    transform.controller
    for transform in env.transform.transforms
    if getattr(transform, "controller", None).__class__.__name__ == "StableSphereResidualSkill"
)

td = env.reset()
trace = []
outcome = "unfinished"
workspace_lower = torch.tensor((0.10, -0.20, 0.02), device=env.device)
workspace_upper = torch.tensor((0.75, 0.48, 0.75), device=env.device)
with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    for step in range(1200):
        td = actor(td)
        policy_action = td["action"].clone()
        tcp_before = td["proprio", "ee_pos"].clone()
        pose_before = td["grasp_guidance", "best_pose_delta"].clone()
        transition, td = env.step_and_maybe_reset(td)
        phase = 0 if residual.phase is None else int(residual.phase[0])
        if step % 20 == 0 or bool(transition["next", "done"][0]):
            outside = bool(
                ((tcp_before[0] < workspace_lower) | (tcp_before[0] > workspace_upper)).any()
            )
            trace.append(
                {
                    "step": step + 1,
                    "phase": residual.baseline.PHASE_NAMES[phase],
                    "policy_action": policy_action[0].cpu().tolist(),
                    "tcp": tcp_before[0].cpu().tolist(),
                    "best_pose_delta": pose_before[0].cpu().tolist(),
                    "outside_workspace": outside,
                }
            )
        if bool(transition["next", "done"][0]):
            names = ("success", "cube_exited_zone", "cube_off_belt", "tcp_escape", "time_out")
            outcome = next(
                (name for name in names if bool(transition["next", "outcome", name][0])),
                "unknown",
            )
            break

report = {
    "checkpoint": str(checkpoint),
    "frames": int(state["frames"]),
    "outcome": outcome,
    "steps": step + 1,
    "final_outside_workspace": trace[-1]["outside_workspace"],
    "trace": trace,
}
output = checkpoint.parent.parent / "diagnostics" / f"{checkpoint.stem}_deterministic_trace.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("JAKA_RESIDUAL_DIAG " + json.dumps({key: value for key, value in report.items() if key != "trace"}))
print(f"TRACE_PATH {output}")
env.close()
os._exit(0)
