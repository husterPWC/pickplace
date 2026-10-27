import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.isaac
REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "pipeline" / "0_state_teacher" / "train.py"
OUTCOMES = ("success", "bowl_exited_zone", "bowl_off_belt", "bowl_tipped", "food_off_table", "time_out")
ITERATIONS = 12
N, ROLLOUT = 32, 16  # 512 frames per iteration


def _lines(stdout, prefix):
    return [json.loads(line[len(prefix):]) for line in stdout.splitlines() if line.startswith(prefix)]


def _train(tmp_path, *overrides, timeout=2400, config_name=None):
    env = {
        **os.environ,
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "FOOD_ROBOT_ARTIFACTS": str(tmp_path / "artifacts"),
        "WANDB_MODE": "offline",
        "WANDB_DIR": str(tmp_path),
        "WANDB_SILENT": "true",
    }
    args = [
        str(TRAIN),
        *(["--config-name", config_name] if config_name else []),
        f"env.num_envs={N}",
        "env.belt.speed=0.3",  # short belt cycle so episodes end within the smoke run
        "env.belt.place_window=0.6",
        f"collector.rollout_steps={ROLLOUT}",
        "loss.ppo_epochs=1",
        "loss.mini_batch_size=256",
        "logger.backend=null",
        f"run.dir={tmp_path / 'run'}",
        f"hydra.run.dir={tmp_path / 'hydra'}",
        *overrides,
    ]
    proc = subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=timeout, env=env, cwd=REPO)
    return proc, proc.stdout[-4000:] + proc.stderr[-4000:]


def test_teacher_trains_logs_terms_and_writes_checkpoints_with_manifests(tmp_path):
    proc, tail = _train(
        tmp_path, f"max_iterations={ITERATIONS}", "checkpoint.interval_frames=3000", "worker.enabled=false"
    )
    assert "PPO_DONE" in proc.stdout, tail

    metrics = _lines(proc.stdout, "METRICS ")
    assert len(metrics) == ITERATIONS, tail
    assert all(math.isfinite(v) for m in metrics for v in m.values() if isinstance(v, (int, float))), "NaN/inf"
    finished = [m for m in metrics if "train/success_rate" in m]
    assert finished, "no episode finished during training"
    for m in finished:
        assert sum(m[f"train/{t}_rate"] for t in OUTCOMES) >= 1.0 - 1e-6
        assert "train/terms/reach_food" in m and "train/terms/food_dropped" in m
    for key in ("train/kl_approx", "train/clip_fraction", "train/entropy", "perf/frames_per_hour"):
        assert key in metrics[-1], key

    run = tmp_path / "run"
    ckpts = sorted((run / "checkpoints").glob("ppo_teacher_*.pt"))
    names = {p.stem for p in ckpts}
    assert "ppo_teacher_final" in names
    assert len(names) >= 3, names  # 3000-frame interval over 6144 frames: two periodic + final
    for p in ckpts:
        m = json.loads(p.with_suffix(".json").read_text())
        assert m["frames"] > 0 and len(m["sha256"]) == 64 and m["config"]["env"]["reward_set"] == "simple_v3b"
        assert m["eval"] is None and m["video"] is None
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["kind"] == "state_teacher" and manifest["stop_reason"] == "total_frames"
    assert set(manifest["checkpoints"]) == names
    assert manifest["config"]["env"]["reward_weights"] == {}


def test_teacher_trains_with_compile_and_shifted_gae(tmp_path):
    proc, tail = _train(
        tmp_path, "max_iterations=4", "checkpoint.interval_frames=0", "worker.enabled=false",
        "compile.compile=true", "loss.shifted_gae=true",
    )
    assert "PPO_DONE" in proc.stdout, tail
    metrics = _lines(proc.stdout, "METRICS ")
    assert len(metrics) == 4, tail
    assert all(math.isfinite(v) for m in metrics for v in m.values() if isinstance(v, (int, float))), "NaN/inf"
    for key in ("train/loss_objective", "train/loss_critic", "train/kl_approx", "perf/frames_per_hour"):
        assert key in metrics[-1], key


def test_teacher_v3c_config_trains(tmp_path):
    """config_v3c is the config that trained the published teacher: the base config plus randomized start poses."""
    proc, tail = _train(
        tmp_path, "max_iterations=4", "checkpoint.interval_frames=0", "worker.enabled=false",
        "init_checkpoint=null", config_name="config_v3c",
    )
    assert "PPO_DONE" in proc.stdout, tail
    assert len(_lines(proc.stdout, "METRICS ")) == 4, tail
    config = json.loads((tmp_path / "run" / "manifest.json").read_text())["config"]
    assert config["env"]["reward_set"] == "simple_v3b" and config["env"]["success_requires_home"] is True
    assert config["env"]["robot_reset"]["position_range"] == [-0.25, 0.25]
    assert config["collector"]["total_frames"] == 300_000_000 and config["logger"]["exp_name"] == "state_teacher_v3c"
    # everything else comes from config.yaml
    assert config["network"]["hidden"] == [512, 256, 128] and config["env"]["action_mode"] == "ee_delta_pose"


def test_teacher_repro_v2_config_trains(tmp_path):
    """The reconstructed first curriculum stage uses placement-only success and the historical reward."""
    proc, tail = _train(
        tmp_path, "max_iterations=4", "checkpoint.interval_frames=0", "worker.enabled=false",
        config_name="config_repro_v2",
    )
    assert "PPO_DONE" in proc.stdout, tail
    config = json.loads((tmp_path / "run" / "manifest.json").read_text())["config"]
    assert config["env"]["reward_set"] == "simple_v2"
    assert config["env"]["success_requires_home"] is False
    assert config["env"]["robot_reset"] == {
        "position_range": [-0.02, 0.02], "velocity_range": [0.0, 0.0]
    }
    assert config["collector"]["total_frames"] == 600_000_000


def test_teacher_c0h_stable_grasp_config_trains(tmp_path):
    """The hold/slip probe builds both stateful rewards and keeps finite PPO losses."""
    proc, tail = _train(
        tmp_path,
        "max_iterations=4",
        "checkpoint.interval_frames=0",
        "worker.enabled=false",
        "init_checkpoint=null",
        config_name="config_moving_cube_c0h_hold_stability",
    )
    assert "PPO_DONE" in proc.stdout, tail
    metrics = _lines(proc.stdout, "METRICS ")
    assert len(metrics) == 4
    assert all(math.isfinite(v) for m in metrics for v in m.values() if isinstance(v, (int, float)))
    config = json.loads((tmp_path / "run" / "manifest.json").read_text())["config"]
    assert config["env"]["grasp_stability"] == {
        "hold_steps": 6, "hold_reward": 3.0, "slip_penalty": 10.0,
    }
    assert config["env"]["reward_weights"]["grasp"] == 1.0
    assert config["loss"]["freeze_actor_scale"] is True


def test_teacher_c0i_staged_hold_config_trains(tmp_path):
    """The true hold-then-lift FSM builds and remains finite under PPO."""
    proc, tail = _train(
        tmp_path,
        "max_iterations=4",
        "checkpoint.interval_frames=0",
        "worker.enabled=false",
        "init_checkpoint=null",
        config_name="config_moving_cube_c0i_staged_hold",
    )
    assert "PPO_DONE" in proc.stdout, tail
    metrics = _lines(proc.stdout, "METRICS ")
    assert len(metrics) == 4
    assert all(math.isfinite(v) for m in metrics for v in m.values() if isinstance(v, (int, float)))
    config = json.loads((tmp_path / "run" / "manifest.json").read_text())["config"]
    assert config["env"]["grasp_stability"] == {
        "hold_steps": 5,
        "hold_reward": 3.0,
        "slip_penalty": 2.0,
        "prelift_max_height": 0.02,
        "premature_lift_penalty": 0.25,
        "require_stable_success": True,
    }
    assert config["loss"]["train_actor_head_only"] is True


def test_teacher_grasp_hold_action_gate_trains(tmp_path):
    """The post-policy hold gate integrates with native Isaac actions and PPO."""
    proc, tail = _train(
        tmp_path,
        "max_iterations=2",
        "checkpoint.interval_frames=0",
        "worker.enabled=false",
        "init_checkpoint=null",
        "env.grasp_hold_gate.hold_steps=5",
        "env.grasp_hold_gate.follow_object_xy=true",
        "env.grasp_hold_gate.freeze_rotation=true",
        config_name="config_moving_cube_c0_lift",
    )
    assert "PPO_DONE" in proc.stdout, tail
    config = json.loads((tmp_path / "run" / "manifest.json").read_text())["config"]
    assert config["env"]["grasp_hold_gate"] == {
        "hold_steps": 5, "max_upward_dz": 0.0, "force_close": True,
        "follow_object_xy": True, "follow_gain": 0.5, "max_follow_speed": 0.15,
        "freeze_rotation": True,
    }


def test_teacher_warm_starts_from_init_checkpoint(tmp_path):
    first, tail = _train(tmp_path / "first", "max_iterations=2", "checkpoint.interval_frames=0", "worker.enabled=false")
    assert "PPO_DONE" in first.stdout, tail
    init_checkpoint = tmp_path / "first" / "run" / "checkpoints" / "ppo_teacher_final.pt"
    assert init_checkpoint.exists()

    second, tail2 = _train(
        tmp_path / "second", "max_iterations=2", "checkpoint.interval_frames=0", "worker.enabled=false",
        f"init_checkpoint={init_checkpoint}",
    )
    assert "PPO_DONE" in second.stdout, tail2
    manifest = json.loads((tmp_path / "second" / "run" / "manifest.json").read_text())
    assert manifest["init_checkpoint"] == str(init_checkpoint)


def test_teacher_resumes_optimizer_frames_and_iteration(tmp_path):
    first, tail = _train(tmp_path / "first", "max_iterations=2", "checkpoint.interval_frames=0", "worker.enabled=false")
    assert "PPO_DONE" in first.stdout, tail
    resume_checkpoint = tmp_path / "first" / "run" / "checkpoints" / "ppo_teacher_final.pt"

    second, tail2 = _train(
        tmp_path / "second", "max_iterations=4", "checkpoint.interval_frames=0", "worker.enabled=false",
        f"resume_checkpoint={resume_checkpoint}",
    )
    assert "PPO_DONE" in second.stdout, tail2
    metrics = _lines(second.stdout, "METRICS ")
    assert [m["iteration"] for m in metrics] == [2, 3], tail2
    assert [m["frames"] for m in metrics] == [3 * N * ROLLOUT, 4 * N * ROLLOUT], tail2
    manifest = json.loads((tmp_path / "second" / "run" / "manifest.json").read_text())
    assert manifest["resume_checkpoint"] == str(resume_checkpoint)
    assert manifest["resume_frames"] == 2 * N * ROLLOUT and manifest["resume_iteration"] == 1
    final = json.loads((tmp_path / "second" / "run" / "checkpoints" / "ppo_teacher_final.json").read_text())
    assert final["frames"] == 4 * N * ROLLOUT and final["iteration"] == 3


def test_teacher_early_stop_saves_final_checkpoint(tmp_path):
    proc, tail = _train(
        tmp_path, "max_iterations=200", "checkpoint.interval_frames=512",
        "worker.enabled=true", "worker.eval_num_envs=16", "worker.render=false",
        "early_stop.success_rate=0.0", "early_stop.consecutive_evals=1",
        timeout=3600,
    )
    assert "PPO_DONE" in proc.stdout, tail
    assert _lines(proc.stdout, "STOP_REASON ")[-1]["reason"] == "early_stop"
    assert len(_lines(proc.stdout, "METRICS ")) < 200
    assert (tmp_path / "run" / "checkpoints" / "ppo_teacher_final.pt").exists()


def test_worker_evaluates_and_renders_every_checkpoint(tmp_path):
    proc, tail = _train(
        tmp_path, "max_iterations=6", "checkpoint.interval_frames=1500",
        "worker.enabled=true", "worker.eval_num_envs=16", "worker.render_seconds=3",
        "logger.backend=wandb", timeout=3600,
    )
    assert "PPO_DONE" in proc.stdout, tail
    ckpts = sorted((tmp_path / "run" / "checkpoints").glob("ppo_teacher_*.pt"))
    assert {p.stem for p in ckpts} >= {"ppo_teacher_1536", "ppo_teacher_3072", "ppo_teacher_final"}
    for p in ckpts:
        m = json.loads(p.with_suffix(".json").read_text())
        assert m["eval"] is not None, (p.name, tail)
        assert 0.0 <= m["eval"]["success_rate"] <= 1.0 and m["eval"]["finished_fraction"] > 0
        assert "terms/reach_food" in m["eval"] and m["eval_num_envs"] == 16
        assert m["video"] == f"{p.stem}.mp4" and (p.parent / m["video"]).stat().st_size > 0
    worker_log = (tmp_path / "run" / "worker.log").read_text()
    assert "EVALUATE_DONE" in worker_log and "RENDER_DONE" in worker_log
    metrics = _lines(proc.stdout, "EVAL_LOGGED ")
    assert {m["checkpoint_frames"] for m in metrics} >= {1536, 3072}
