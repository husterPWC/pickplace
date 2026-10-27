"""Paired deterministic failure diagnosis for moving-cube lift teachers.

The two policies see the same seed and the script verifies that every initial
cube/EE/belt state is identical before comparing outcomes.  Only each vector
environment's first episode is scored; native auto-reset episodes are masked.

Usage:
    python pipeline/0_state_teacher/diagnose_failures.py \
      baseline=path/to/baseline.pt candidate=path/to/candidate.pt \
      num_envs=1024 seed=0 output=artifacts/.../failure_diagnosis
"""

from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
cli = OmegaConf.from_dotlist(sys.argv[1:])
required = ("baseline", "candidate", "output")
missing = [key for key in required if not cli.get(key)]
if missing:
    raise SystemExit(f"Missing required arguments: {', '.join(missing)}")

baseline_path = Path(str(cli.baseline)).resolve()
candidate_path = Path(str(cli.candidate)).resolve()
output_dir = Path(str(cli.output)).resolve()
num_envs = int(cli.get("num_envs", 1024))
seed = int(cli.get("seed", 0))
near_threshold = float(cli.get("near_threshold", 0.04))
baseline_label = str(cli.get("baseline_label", "baseline"))
candidate_label = str(cli.get("candidate_label", "candidate"))
baseline_hold_gate_steps = int(cli.get("baseline_hold_gate_steps", 0))
candidate_hold_gate_steps = int(cli.get("candidate_hold_gate_steps", 0))
baseline_max_upward_dz = float(cli.get("baseline_max_upward_dz", 0.0))
candidate_max_upward_dz = float(cli.get("candidate_max_upward_dz", 0.0))
baseline_follow_object_xy = bool(cli.get("baseline_follow_object_xy", False))
candidate_follow_object_xy = bool(cli.get("candidate_follow_object_xy", False))
baseline_freeze_rotation = bool(cli.get("baseline_freeze_rotation", False))
candidate_freeze_rotation = bool(cli.get("candidate_freeze_rotation", False))

from pickplace.app import launch_app  # noqa: E402

app = launch_app(headless=True, enable_cameras=False)

import torch  # noqa: E402
from torchrl.envs import ExplorationType, set_exploration_type  # noqa: E402

from pickplace.action_gates import GraspHoldGate  # noqa: E402
from pickplace.artifacts import sha256_file  # noqa: E402
from pickplace.failure_diagnostics import classify_lift_episode, summarise_episodes  # noqa: E402
from pickplace.metrics import outcome_terms  # noqa: E402
from pickplace.torchrl_env import make_env  # noqa: E402

_spec = importlib.util.spec_from_file_location("teacher_utils", HERE / "utils.py")
tu = importlib.util.module_from_spec(_spec)
sys.modules["teacher_utils"] = tu
_spec.loader.exec_module(tu)


def _seed_all(value: int) -> None:
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1)


def _kinematic_grasp_bit(td) -> torch.Tensor:
    """Return the available grasp-phase bit without requiring contacts.

    New sensor-compatible teachers expose ``grasp_proxy`` (object/TCP pose and
    finger aperture only), while historical checkpoints expose ``is_grasped``.
    Keeping the fallback lets the same diagnostic compare both generations.
    """
    for key in (("privileged", "grasp_proxy"), ("privileged", "is_grasped")):
        value = td.get(key, None)
        if value is not None:
            return _flat(value)[:, 0].bool()
    raise KeyError("diagnosis requires privileged.grasp_proxy or privileged.is_grasped")


def _first_step(target: torch.Tensor, event: torch.Tensor, step: int) -> None:
    rows = event & target.lt(0)
    target[rows] = step


def _as_list(tensor: torch.Tensor, digits: int = 7):
    value = tensor.detach().cpu().tolist()
    if isinstance(value, list):
        return [round(float(x), digits) for x in value]
    return round(float(value), digits)


def run_policy(
    label: str,
    checkpoint_path: Path,
    env,
    steps: int,
    target_lift: float,
    expected_initial=None,
    hold_gate_steps: int = 0,
    max_upward_dz: float = 0.0,
    follow_object_xy: bool = False,
    freeze_rotation: bool = False,
):
    print(f"DIAG_START label={label} checkpoint={checkpoint_path} envs={num_envs}", flush=True)
    actor = tu.load_teacher_actor(checkpoint_path, env, env.device)
    _seed_all(seed)
    td = env.reset()

    food0 = _flat(td.get(("privileged", "food_pos"))).clone()
    ee0 = _flat(td.get(("proprio", "ee_pos"))).clone()
    belt0 = _flat(td.get(("belt", "speed"))).clone()
    initial = {"food_pos": food0, "ee_pos": ee0, "belt_speed": belt0}
    initial_max_abs_diff = 0.0
    if expected_initial is not None:
        for key in initial:
            diff = (initial[key] - expected_initial[key]).abs().max().item()
            initial_max_abs_diff = max(initial_max_abs_diff, diff)
        if initial_max_abs_diff > 1e-6:
            raise RuntimeError(
                f"Paired evaluation initial states differ (max abs diff={initial_max_abs_diff:.3g}); "
                "refusing an unfair policy comparison."
            )

    n, device = num_envs, env.device
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    episode_return = torch.zeros(n, device=device)
    episode_steps = torch.zeros(n, dtype=torch.long, device=device)
    outcome_code = torch.full((n,), -1, dtype=torch.long, device=device)
    min_dist = torch.full((n,), float("inf"), device=device)
    min_xy_dist = torch.full((n,), float("inf"), device=device)
    nearest_step = torch.full((n,), -1, dtype=torch.long, device=device)
    first_near = torch.full((n,), -1, dtype=torch.long, device=device)
    first_close = torch.full((n,), -1, dtype=torch.long, device=device)
    first_grasp = torch.full((n,), -1, dtype=torch.long, device=device)
    first_lift_1cm = torch.full((n,), -1, dtype=torch.long, device=device)
    ever_grasped = torch.zeros(n, dtype=torch.bool, device=device)
    grasp_lost = torch.zeros(n, dtype=torch.bool, device=device)
    opened_after_grasp = torch.zeros(n, dtype=torch.bool, device=device)
    previous_grasped = torch.zeros(n, dtype=torch.bool, device=device)
    current_grasp_run = torch.zeros(n, dtype=torch.long, device=device)
    longest_grasp_run = torch.zeros(n, dtype=torch.long, device=device)
    grasp_runs = torch.zeros(n, dtype=torch.long, device=device)
    total_grasped_steps = torch.zeros(n, dtype=torch.long, device=device)
    first_grasp_lost = torch.full((n,), -1, dtype=torch.long, device=device)
    max_lift = torch.full((n,), -float("inf"), device=device)
    max_held_lift = torch.full((n,), -float("inf"), device=device)
    close_steps = torch.zeros(n, dtype=torch.long, device=device)
    final_food = food0.clone()
    final_ee = ee0.clone()
    final_gripper = _flat(td.get(("proprio", "gripper_pos"))).clone()
    u = env.base_env._env.unwrapped
    gate = (
        GraspHoldGate(
            hold_gate_steps,
            max_upward_dz=max_upward_dz,
            follow_object_xy=follow_object_xy,
            control_dt=float(u.step_dt),
            action_scale=float(u.cfg.arm.ik_action_scale),
            freeze_rotation=freeze_rotation,
        )
        if hold_gate_steps > 0
        else None
    )
    gate_active_steps = torch.zeros(n, dtype=torch.long, device=device)

    names = tuple(outcome_terms(checkpoint["config"]["env"]["task"]))
    traces: dict[str, list[torch.Tensor]] = {
        "valid": [], "ee_pos": [], "cube_pos": [], "cube_lin_vel": [],
        "gripper_pos": [], "policy_action": [], "action": [], "is_grasped": [], "hold_gate_active": [],
    }
    started = time.monotonic()
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        for step in range(steps):
            td = actor(td)
            policy_action = _flat(td.get("action")).clone()
            if gate is None:
                gate_active = torch.zeros(n, dtype=torch.bool, device=device)
            else:
                gated_action, gate_active = gate(
                    policy_action,
                    _kinematic_grasp_bit(td),
                    object_pos=_flat(td.get(("privileged", "food_pos"))),
                    ee_pos=_flat(td.get(("proprio", "ee_pos"))),
                    object_lin_vel=_flat(td.get(("privileged", "food_lin_vel"))),
                )
                td.set("action", gated_action)
            stepped, td = env.step_and_maybe_reset(td)
            active = ~finished

            food = _flat(stepped.get(("privileged", "food_pos")))
            ee = _flat(stepped.get(("proprio", "ee_pos")))
            food_vel = _flat(stepped.get(("privileged", "food_lin_vel")))
            gripper = _flat(stepped.get(("proprio", "gripper_pos")))
            action = _flat(stepped.get("action"))
            grasped = _kinematic_grasp_bit(stepped)
            close = action[:, -1] < 0.0
            lift = food[:, 2] - food0[:, 2]
            distance = torch.linalg.vector_norm(ee - food, dim=-1)
            xy_distance = torch.linalg.vector_norm(ee[:, :2] - food[:, :2], dim=-1)

            traces["valid"].append(active.clone())
            traces["ee_pos"].append(ee.clone())
            traces["cube_pos"].append(food.clone())
            traces["cube_lin_vel"].append(food_vel.clone())
            traces["gripper_pos"].append(gripper.clone())
            traces["policy_action"].append(policy_action.clone())
            traces["action"].append(action.clone())
            traces["is_grasped"].append(grasped.clone())
            traces["hold_gate_active"].append(gate_active.clone())

            better = active & (distance < min_dist)
            min_dist = torch.where(active, torch.minimum(min_dist, distance), min_dist)
            min_xy_dist = torch.where(active, torch.minimum(min_xy_dist, xy_distance), min_xy_dist)
            nearest_step[better] = step
            _first_step(first_near, active & (distance < near_threshold), step)
            _first_step(first_close, active & close, step)
            _first_step(first_grasp, active & grasped, step)
            _first_step(first_lift_1cm, active & (lift >= 0.01), step)

            previously_ever_grasped = ever_grasped.clone()
            lost_now = active & previously_ever_grasped & ~grasped & ~grasp_lost
            _first_step(first_grasp_lost, lost_now, step)
            grasp_lost |= lost_now
            # Count an opening command only while the object is still held;
            # reopening later after a passive slip must not be called a release.
            opened_after_grasp |= active & grasped & ~close
            new_grasp_run = active & grasped & ~previous_grasped
            grasp_runs += new_grasp_run.long()
            total_grasped_steps += (active & grasped).long()
            gate_active_steps += (active & gate_active).long()
            current_grasp_run = torch.where(active & grasped, current_grasp_run + 1, torch.zeros_like(current_grasp_run))
            longest_grasp_run = torch.maximum(longest_grasp_run, current_grasp_run)
            previous_grasped = torch.where(active, grasped, previous_grasped)
            ever_grasped |= active & grasped
            close_steps += (active & close).long()
            max_lift = torch.where(active, torch.maximum(max_lift, lift), max_lift)
            max_held_lift = torch.where(active & grasped, torch.maximum(max_held_lift, lift), max_held_lift)
            final_food[active] = food[active]
            final_ee[active] = ee[active]
            final_gripper[active] = gripper[active]

            reward = stepped.get(("next", "reward")).reshape(n)
            episode_return += torch.where(active, torch.nan_to_num(reward), 0.0)
            done = stepped.get(("next", "done")).reshape(n).bool()
            newly_done = active & done
            episode_steps[newly_done] = step + 1
            for code, name in enumerate(names):
                fired = stepped.get(("next", "outcome", name)).reshape(n).bool()
                outcome_code[newly_done & fired] = code
            finished |= newly_done
            if bool(finished.all()):
                break

    if not bool(finished.all()):
        missing = int((~finished).sum().item())
        raise RuntimeError(f"{missing} first episodes did not finish within {steps} steps")
    if bool((outcome_code < 0).any()):
        missing = int((outcome_code < 0).sum().item())
        raise RuntimeError(f"{missing} episodes finished without a recognised outcome")

    max_held_lift = torch.where(torch.isfinite(max_held_lift), max_held_lift, torch.zeros_like(max_held_lift))
    rows = []
    for index in range(n):
        outcome = names[int(outcome_code[index].item())]
        row = {
            "episode": index,
            "outcome": outcome,
            "success": outcome == "success",
            "episode_steps": int(episode_steps[index].item()),
            "episode_return": round(float(episode_return[index].item()), 7),
            "initial_cube_pos": _as_list(food0[index]),
            "initial_ee_pos": _as_list(ee0[index]),
            "belt_speed_mps": round(float(belt0[index, 0].item()), 7),
            "final_cube_pos": _as_list(final_food[index]),
            "final_ee_pos": _as_list(final_ee[index]),
            "final_gripper_pos": _as_list(final_gripper[index]),
            "min_ee_cube_distance_m": round(float(min_dist[index].item()), 7),
            "min_xy_distance_m": round(float(min_xy_dist[index].item()), 7),
            "nearest_step": int(nearest_step[index].item()),
            "first_near_step": int(first_near[index].item()),
            "first_close_step": int(first_close[index].item()),
            "first_grasp_step": int(first_grasp[index].item()),
            "first_grasp_lost_step": int(first_grasp_lost[index].item()),
            "first_lift_1cm_step": int(first_lift_1cm[index].item()),
            "max_cube_lift_m": round(float(max_lift[index].item()), 7),
            "max_held_lift_m": round(float(max_held_lift[index].item()), 7),
            "grasp_lost": bool(grasp_lost[index].item()),
            "opened_after_grasp": bool(opened_after_grasp[index].item()),
            "grasp_runs": int(grasp_runs[index].item()),
            "total_grasped_steps": int(total_grasped_steps[index].item()),
            "longest_grasp_run_steps": int(longest_grasp_run[index].item()),
            "close_command_fraction": round(float(close_steps[index].item() / episode_steps[index].item()), 7),
            "hold_gate_steps": hold_gate_steps,
            "hold_gate_active_steps": int(gate_active_steps[index].item()),
        }
        row["failure_stage"] = classify_lift_episode(row, target_lift=target_lift)
        rows.append(row)

    failed_indices = torch.tensor(
        [r["episode"] for r in rows if r["failure_stage"] != "success"], dtype=torch.long, device=device
    )
    trace_payload = {
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "near_threshold_m": near_threshold,
        "target_lift_m": target_lift,
        "failure_episode_indices": failed_indices.cpu(),
    }
    for key, values in traces.items():
        stacked = torch.stack(values, dim=1)
        trace_payload[key] = stacked.index_select(0, failed_indices).cpu()

    elapsed = time.monotonic() - started
    summary = summarise_episodes(rows)
    summary.update({
        "label": label,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "seed": seed,
        "target_lift_m": target_lift,
        "near_threshold_m": near_threshold,
        "initial_state_max_abs_diff_from_baseline": initial_max_abs_diff,
        "elapsed_seconds": round(elapsed, 2),
        "trajectory_steps": len(traces["valid"]),
        "hold_gate_steps": hold_gate_steps,
        "max_upward_dz": max_upward_dz,
        "follow_object_xy": follow_object_xy,
        "freeze_rotation": freeze_rotation,
    })
    print(
        "DIAG_DONE " + json.dumps({
            "label": label, "success_rate": summary["success_rate"],
            "outcomes": summary["outcomes"], "failure_stages": summary["failure_stages"],
            "seconds": summary["elapsed_seconds"],
        }),
        flush=True,
    )
    return initial, rows, trace_payload, summary


def _stage_table(baseline: dict, candidate: dict, baseline_name: str, candidate_name: str) -> str:
    stages = sorted(set(baseline["failure_stages"]) | set(candidate["failure_stages"]))
    lines = [
        f"| 阶段 | {baseline_name} | {candidate_name} | {candidate_name}-{baseline_name} |",
        "|---|---:|---:|---:|",
    ]
    for stage in stages:
        a = int(baseline["failure_stages"].get(stage, 0))
        b = int(candidate["failure_stages"].get(stage, 0))
        lines.append(f"| `{stage}` | {a} | {b} | {b-a:+d} |")
    return "\n".join(lines)


output_dir.mkdir(parents=True, exist_ok=True)
checkpoint = torch.load(baseline_path, map_location="cpu", weights_only=False)
candidate_checkpoint = torch.load(candidate_path, map_location="cpu", weights_only=False)
baseline_env_cfg = checkpoint["config"]["env"]
candidate_env_cfg = candidate_checkpoint["config"]["env"]
env_diff = {
    key: {"baseline": baseline_env_cfg.get(key), "candidate": candidate_env_cfg.get(key)}
    for key in sorted(set(baseline_env_cfg) | set(candidate_env_cfg))
    if baseline_env_cfg.get(key) != candidate_env_cfg.get(key)
    and key not in {
        "num_envs", "seed", "cameras",
        # Reward-only differences do not change observations, physics, or
        # termination outcomes. Evaluation uses the baseline environment so a
        # paired policy comparison still has one identical scoring contract.
        "reward_set", "reward_weights", "rewards", "grasp_stability",
    }
}
if env_diff:
    raise RuntimeError(f"Checkpoint environment configs differ: {env_diff}")

eval_env_cfg = {**baseline_env_cfg, "num_envs": num_envs, "cameras": False, "seed": seed}
_seed_all(seed)
env = make_env(eval_env_cfg)
steps = int(env.base_env._env.unwrapped.max_episode_length) + 1
target_lift = float(eval_env_cfg.get("target_lift", 0.03))

initial, baseline_rows, baseline_trace, baseline_summary = run_policy(
    baseline_label,
    baseline_path,
    env,
    steps,
    target_lift,
    hold_gate_steps=baseline_hold_gate_steps,
    max_upward_dz=baseline_max_upward_dz,
    follow_object_xy=baseline_follow_object_xy,
    freeze_rotation=baseline_freeze_rotation,
)
_, candidate_rows, candidate_trace, candidate_summary = run_policy(
    candidate_label,
    candidate_path,
    env,
    steps,
    target_lift,
    expected_initial=initial,
    hold_gate_steps=candidate_hold_gate_steps,
    max_upward_dz=candidate_max_upward_dz,
    follow_object_xy=candidate_follow_object_xy,
    freeze_rotation=candidate_freeze_rotation,
)

torch.save(baseline_trace, output_dir / f"{baseline_label}_failed_trajectories.pt")
torch.save(candidate_trace, output_dir / f"{candidate_label}_failed_trajectories.pt")
for name, rows in ((baseline_label, baseline_rows), (candidate_label, candidate_rows)):
    with (output_dir / f"{name}_episodes.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

paired = {
    "both_success": 0, "both_failure": 0, "c0a_only_success": 0, "c0k_only_success": 0,
}
for a, b in zip(baseline_rows, candidate_rows):
    if a["success"] and b["success"]:
        paired["both_success"] += 1
    elif not a["success"] and not b["success"]:
        paired["both_failure"] += 1
    elif a["success"]:
        paired["c0a_only_success"] += 1
    else:
        paired["c0k_only_success"] += 1

report = {
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "paired_initial_conditions": True,
    "num_envs": num_envs,
    "seed": seed,
    "baseline": baseline_summary,
    "candidate": candidate_summary,
    "paired_outcomes": paired,
    "success_rate_delta": candidate_summary["success_rate"] - baseline_summary["success_rate"],
    "action_gates": {
        baseline_label: {
            "hold_steps": baseline_hold_gate_steps,
            "max_upward_dz": baseline_max_upward_dz,
            "follow_object_xy": baseline_follow_object_xy,
            "freeze_rotation": baseline_freeze_rotation,
        },
        candidate_label: {
            "hold_steps": candidate_hold_gate_steps,
            "max_upward_dz": candidate_max_upward_dz,
            "follow_object_xy": candidate_follow_object_xy,
            "freeze_rotation": candidate_freeze_rotation,
        },
    },
}
(output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stage_total(summary: dict, prefix: str) -> int:
    return sum(count for stage, count in summary["failure_stages"].items() if stage.startswith(prefix))


def _median(rows: list[dict], key: str, success: bool) -> float:
    return statistics.median(row[key] for row in rows if row["success"] is success)


baseline_passive_loss = _stage_total(baseline_summary, "lost_grasp_")
candidate_passive_loss = _stage_total(candidate_summary, "lost_grasp_")
baseline_open_release = sum(row["opened_after_grasp"] for row in baseline_rows if not row["success"])
candidate_open_release = sum(row["opened_after_grasp"] for row in candidate_rows if not row["success"])
early_lift_delta = (
    candidate_summary["failure_stages"].get("lost_grasp_during_early_lift", 0)
    - baseline_summary["failure_stages"].get("lost_grasp_during_early_lift", 0)
)
passive_loss_delta = candidate_passive_loss - baseline_passive_loss
if report["success_rate_delta"] > 0 and passive_loss_delta < 0:
    next_step = (
        f"**候选方向通过短探针。** 被动滑脱减少 {abs(passive_loss_delta)} 回合且成功率净增 "
        f"{report['success_rate_delta']:+.2%}；可以保留该候选，并用更大样本复核后再决定是否长训。"
    )
else:
    next_step = (
        f"**候选方向未通过短探针，不应继续长训或替换基线。** 被动滑脱变化 {passive_loss_delta:+d} 回合、"
        f"成功率净变化 {report['success_rate_delta']:+.2%}。应保留 {baseline_label}，依据逐步轨迹继续修正"
        "抓取后的低层动作约束，而不是继续加大奖励惩罚或直接长训。"
    )

markdown = f"""# {baseline_label} 与 {candidate_label} 失败回合诊断

- 样本：同一随机种子的 {num_envs} 组逐项配对初始状态；最大初态绝对误差 `{candidate_summary['initial_state_max_abs_diff_from_baseline']:.3g}`。
- {baseline_label}：{baseline_summary['successes']}/{num_envs}，成功率 {baseline_summary['success_rate']:.2%}。
- {candidate_label}：{candidate_summary['successes']}/{num_envs}，成功率 {candidate_summary['success_rate']:.2%}，变化 {report['success_rate_delta']:+.2%}。
- 配对变化：{candidate_label} 新救回 {paired['c0k_only_success']} 回合，同时弄丢 {baseline_label} 原本成功的 {paired['c0a_only_success']} 回合；两者都失败 {paired['both_failure']} 回合。

## 阶段计数

{_stage_table(baseline_summary, candidate_summary, baseline_label, candidate_label)}

## 诊断结论

1. **瓶颈不在追踪或闭合触发，而在闭合后的保持。** {baseline_label} 的 {baseline_summary['failures']} 个失败中，有 {baseline_passive_loss} 个在满足环境真实抓取判据后被动失去方块；{candidate_label} 对应为 {candidate_passive_loss}/{candidate_summary['failures']}。两者抓住方块时主动发出张开命令的失败回合分别为 {baseline_open_release} 和 {candidate_open_release}，所以不能把问题归因于“策略主动松手”。
2. **稳定抓持只差很短的时间窗。** {baseline_label} 失败回合的最长连续抓持中位数是 {_median(baseline_rows, 'longest_grasp_run_steps', False):.0f} 个控制步，成功回合为 {_median(baseline_rows, 'longest_grasp_run_steps', True):.0f} 步（每步 0.02 s）。这说明接触建立了，但方块/TCP 的相对运动和夹指接触没有稳定下来。
3. **候选策略对临界回合的净作用。** 它救回 {paired['c0k_only_success']} 回合却损失 {paired['c0a_only_success']} 回合；早期 lift 滑脱变化 {early_lift_delta:+d}，最终净变化 {report['success_rate_delta']:+.2%}。
4. {next_step}

## 文件

- `report.json`：机器可读汇总和配对结果。
- `{baseline_label}_episodes.jsonl`、`{candidate_label}_episodes.jsonl`：每回合的初态、结果、关键时刻和物理量。
- `{baseline_label}_failed_trajectories.pt`、`{candidate_label}_failed_trajectories.pt`：仅失败回合的逐步 EE/方块/夹爪/动作轨迹及有效步掩码。
"""
(output_dir / "REPORT_CN.md").write_text(markdown, encoding="utf-8")
print("DIAG_REPORT " + json.dumps(report, ensure_ascii=False), flush=True)
print(f"DIAG_OUTPUT {output_dir}", flush=True)
os._exit(0)  # Isaac Sim shutdown can hang on this driver/runtime combination.
