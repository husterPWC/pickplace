"""Diagnose the contact-free close-to-lift bridge on first episodes.

The policy remains speed/contact/force free.  This evaluator may inspect its
position-history observation and outcome flags to explain where the strict
bridge fails; none of these diagnostics are fed back to the policy.

Usage:
    python pipeline/0_state_teacher/diagnose_close_lift_bridge.py \
      checkpoint=path/to/ppo_teacher.pt [env_checkpoint=path/to/c0w.pt] \
      [num_envs=256] [seed=0] [output=path/to/report.json]
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
cli = OmegaConf.from_dotlist(sys.argv[1:])
if not cli.get("checkpoint"):
    raise SystemExit("Pass checkpoint=<path.pt>.")
checkpoint_path = Path(str(cli.checkpoint)).resolve()
env_checkpoint_path = Path(str(cli.get("env_checkpoint", checkpoint_path))).resolve()
num_envs = int(cli.get("num_envs", 256))
seed = int(cli.get("seed", 0))
output = Path(str(cli.output)).resolve() if cli.get("output") else None

from pickplace.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=False)

import torch  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from pickplace.metrics import outcome_terms  # noqa: E402
from pickplace.torchrl_env import make_env  # noqa: E402

spec = importlib.util.spec_from_file_location("teacher_utils", HERE / "utils.py")
teacher_utils = importlib.util.module_from_spec(spec)
sys.modules["teacher_utils"] = teacher_utils
spec.loader.exec_module(teacher_utils)


def flat(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1)


env_checkpoint = torch.load(env_checkpoint_path, map_location="cpu", weights_only=False)
env_cfg = dict(env_checkpoint["config"]["env"])
bridge = dict(env_cfg.get("geometric_close_lift_bridge") or {})
if not bridge.get("enabled"):
    raise RuntimeError("env_checkpoint does not enable geometric_close_lift_bridge")
env_cfg.update(num_envs=num_envs, cameras=False, seed=seed)
torch.manual_seed(seed)
env = make_env(env_cfg)
actor = teacher_utils.load_teacher_actor(checkpoint_path, env, env.device)
steps = int(env.base_env._env.unwrapped.max_episode_length) + 1

distance_threshold = float(bridge["distance_threshold"])
minimum = float(bridge["min_close_fraction"])
maximum = float(bridge["max_close_fraction"])
prelift_max_height = float(bridge["prelift_max_height"])
target_lift = float(bridge["target_lift"])
max_drift = float(bridge["max_relative_drift"])
close_steps_needed = int(bridge["close_steps"])
success_steps_needed = int(bridge["success_steps"])
u = env.base_env._env.unwrapped
rest_z = float(u.cfg.belt.plate_top_z + u.cfg.food.item_radius)
open_pos = float(u.cfg.arm.gripper_open)
closed_pos = float(u.cfg.arm.gripper_closed)
span = max(abs(open_pos - closed_pos), 1.0e-6)

n, device = num_envs, env.device
finished = torch.zeros(n, dtype=torch.bool, device=device)
success = torch.zeros(n, dtype=torch.bool, device=device)
close_count = torch.zeros(n, dtype=torch.long, device=device)
lift_count = torch.zeros(n, dtype=torch.long, device=device)
latched = torch.zeros(n, dtype=torch.bool, device=device)
ever_near = torch.zeros(n, dtype=torch.bool, device=device)
ever_aligned_close = torch.zeros(n, dtype=torch.bool, device=device)
ever_latched = torch.zeros(n, dtype=torch.bool, device=device)
lost_distance = torch.zeros(n, dtype=torch.bool, device=device)
lost_aperture = torch.zeros(n, dtype=torch.bool, device=device)
lost_drift = torch.zeros(n, dtype=torch.bool, device=device)
reference_offset = torch.zeros(n, 3, device=device)
max_lift = torch.full((n,), -float("inf"), device=device)
max_coupled_lift = torch.zeros(n, device=device)
post_latch_dz_sum = torch.zeros(n, device=device)
post_latch_dz_max = torch.full((n,), -float("inf"), device=device)
post_latch_steps = torch.zeros(n, dtype=torch.long, device=device)
post_latch_positive_dz = torch.zeros(n, dtype=torch.long, device=device)
outcome = ["unfinished"] * n
names = tuple(outcome_terms(env_cfg["task"]))

td = env.reset()
with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
    for _ in range(steps):
        active = ~finished
        history = td.get(("position_history", "position_action"))
        offset = history[:, -1, :3]
        ee = flat(td.get(("proprio", "ee_pos")))
        food = ee + offset
        fingers = flat(td.get(("proprio", "gripper_pos")))
        distance = torch.linalg.vector_norm(offset, dim=-1)
        lift = (food[:, 2] - rest_z).clamp_min(0.0)
        close_fraction = (1.0 - (fingers.mean(-1) - closed_pos) / span).clamp(0.0, 1.0)
        aperture_ok = (close_fraction >= minimum) & (close_fraction <= maximum)
        near = distance <= distance_threshold
        aligned_close = near & aperture_ok
        acquiring = active & aligned_close & (lift <= prelift_max_height) & ~latched
        close_count = torch.where(
            latched,
            close_count,
            torch.where(acquiring, close_count + 1, torch.zeros_like(close_count)),
        )
        newly_latched = active & ~latched & (close_count >= close_steps_needed)
        reference_offset = torch.where(newly_latched[:, None], offset, reference_offset)
        latched |= newly_latched
        ever_latched |= newly_latched
        ever_near |= active & near
        ever_aligned_close |= active & aligned_close
        max_lift = torch.where(active, torch.maximum(max_lift, lift), max_lift)

        drift = torch.linalg.vector_norm(offset - reference_offset, dim=-1)
        coupled = aligned_close & (drift <= max_drift)
        max_coupled_lift = torch.where(
            active & latched & coupled,
            torch.maximum(max_coupled_lift, lift),
            max_coupled_lift,
        )
        lost = active & latched & ~coupled
        lost_distance |= lost & ~near
        lost_aperture |= lost & ~aperture_ok
        lost_drift |= lost & near & aperture_ok & (drift > max_drift)
        latched &= ~lost
        close_count = torch.where(lost, torch.zeros_like(close_count), close_count)
        lifted = active & latched & coupled & (lift >= target_lift)
        lift_count = torch.where(lifted, lift_count + 1, torch.zeros_like(lift_count))

        td = actor(td)
        action = flat(td.get("action"))
        post_latch_dz_sum += torch.where(active & latched, action[:, 2], 0.0)
        post_latch_dz_max = torch.where(
            active & latched, torch.maximum(post_latch_dz_max, action[:, 2]), post_latch_dz_max
        )
        post_latch_steps += (active & latched).long()
        post_latch_positive_dz += (active & latched & (action[:, 2] > 0)).long()

        stepped, td = env.step_and_maybe_reset(td)
        done = flat(stepped.get(("next", "done")))[:, 0].bool()
        newly_done = active & done
        for name in names:
            fired = flat(stepped.get(("next", "outcome", name)))[:, 0].bool()
            indices = (newly_done & fired).nonzero(as_tuple=False).flatten().tolist()
            for index in indices:
                outcome[index] = name
            if name == "success":
                success |= newly_done & fired
        finished |= newly_done
        if bool(finished.all()):
            break

max_lift = torch.where(torch.isfinite(max_lift), max_lift, torch.zeros_like(max_lift))
post_latch_dz_max = torch.where(
    torch.isfinite(post_latch_dz_max), post_latch_dz_max, torch.zeros_like(post_latch_dz_max)
)
categories = []
for index in range(n):
    if bool(success[index]):
        category = "success"
    elif not bool(ever_near[index]):
        category = "never_near_cube"
    elif not bool(ever_aligned_close[index]):
        category = "near_but_no_valid_aperture"
    elif not bool(ever_latched[index]):
        category = "valid_close_not_held_5_steps"
    elif float(max_coupled_lift[index]) < 0.001:
        category = "latched_but_no_1mm_lift"
    elif float(max_coupled_lift[index]) < 0.003:
        category = "coupled_lift_1_to_3mm"
    elif float(max_coupled_lift[index]) < target_lift:
        category = "coupled_lift_3_to_5mm"
    else:
        category = "reached_5mm_but_not_held_5_steps"
    categories.append(category)

latched_rows = ever_latched
latched_count = int(latched_rows.sum().item())
report = {
    "checkpoint": str(checkpoint_path),
    "env_checkpoint": str(env_checkpoint_path),
    "num_envs": n,
    "seed": seed,
    "bridge_contract": bridge,
    "successes": int(success.sum().item()),
    "success_rate": float(success.float().mean().item()),
    "outcomes": dict(sorted(Counter(outcome).items())),
    "failure_stages": dict(sorted(Counter(categories).items())),
    "phase_rates": {
        "ever_near": float(ever_near.float().mean().item()),
        "ever_valid_close": float(ever_aligned_close.float().mean().item()),
        "ever_stable_close_latched": float(ever_latched.float().mean().item()),
        "ever_coupled_lift_1mm": float((max_coupled_lift >= 0.001).float().mean().item()),
        "ever_coupled_lift_3mm": float((max_coupled_lift >= 0.003).float().mean().item()),
        "ever_coupled_lift_5mm": float((max_coupled_lift >= target_lift).float().mean().item()),
    },
    "latch_loss_rates_among_latched": {
        "distance": float(lost_distance[latched_rows].float().mean().item()) if latched_count else 0.0,
        "aperture": float(lost_aperture[latched_rows].float().mean().item()) if latched_count else 0.0,
        "relative_drift": float(lost_drift[latched_rows].float().mean().item()) if latched_count else 0.0,
    },
    "post_latch_action_dz": {
        "episodes": latched_count,
        "mean": float(
            (post_latch_dz_sum[latched_rows] / post_latch_steps[latched_rows].clamp_min(1)).mean().item()
        ) if latched_count else 0.0,
        "mean_positive_fraction": float(
            (post_latch_positive_dz[latched_rows].float() / post_latch_steps[latched_rows].clamp_min(1)).mean().item()
        ) if latched_count else 0.0,
        "mean_episode_max": float(post_latch_dz_max[latched_rows].mean().item()) if latched_count else 0.0,
    },
    "lift_m": {
        "mean_episode_max_any": float(max_lift.mean().item()),
        "mean_episode_max_coupled": float(max_coupled_lift.mean().item()),
        "max_coupled": float(max_coupled_lift.max().item()),
    },
}
if output is not None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("BRIDGE_DIAG " + json.dumps(report, ensure_ascii=False), flush=True)
os._exit(0)
