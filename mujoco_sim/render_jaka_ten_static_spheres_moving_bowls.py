#!/usr/bin/env python3
"""Render ten static balls sorted into ten already-moving bowls."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
from pathlib import Path

if os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1":
    raise RuntimeError("render requires LIBGL_ALWAYS_SOFTWARE=1")
if os.environ.get("MUJOCO_GL") != "glfw":
    raise RuntimeError("render requires MUJOCO_GL=glfw under Xvfb")
if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
    raise RuntimeError("MuJoCo render must not use a GPU")

import mujoco
import numpy as np

from mujoco_sim.env import POLICY_DT, ROOT
from mujoco_sim.evaluate_jaka_ten_static_spheres_moving_bowls import (
    DEFAULT_CHECKPOINT,
    pair_snapshot,
)
from mujoco_sim.jaka_moving_sphere import JakaMovingSphereMuJoCo
from mujoco_sim.moving_sphere import GraspPointGRUActor
from mujoco_sim.policy_runner import parse_cli


DEFAULT_OUTPUT = ROOT / (
    "artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/mujoco/"
    "ten_static_spheres_ten_visible_moving_bowls_speed010_"
    "checkpoint_closed_loop_full_scene_cpu.mp4"
)


def configure_upstream_camera(camera: mujoco.MjvCamera) -> None:
    camera.lookat[:] = (-3.30, 0.23, 0.10)
    camera.distance = 8.25
    camera.azimuth = 90
    camera.elevation = -31


def configure_supply_camera(camera: mujoco.MjvCamera) -> None:
    camera.lookat[:] = (0.44, 0.06, 0.13)
    camera.distance = 1.10
    camera.azimuth = 132
    camera.elevation = -34


def configure_workcell_camera(camera: mujoco.MjvCamera) -> None:
    camera.lookat[:] = (0.40, 0.16, 0.22)
    camera.distance = 1.52
    camera.azimuth = 132
    camera.elevation = -24


def configure_downstream_camera(camera: mujoco.MjvCamera) -> None:
    camera.lookat[:] = (3.75, 0.26, 0.10)
    camera.distance = 8.55
    camera.azimuth = 90
    camera.elevation = -31


def main() -> None:
    cli = parse_cli()
    output = Path(cli.get("out", DEFAULT_OUTPUT)).resolve()
    checkpoint = Path(cli.get("checkpoint", DEFAULT_CHECKPOINT)).resolve()
    seed = int(cli.get("seed", 8300))
    num_items = int(cli.get("num_items", 10))
    bowl_speed = float(cli.get("bowl_speed", 0.10))
    spacing = float(cli.get("spacing", 0.75))
    max_seconds_per_call = float(cli.get("max_seconds_per_call", 12.0))
    width = int(cli.get("width", 960))
    height = int(cli.get("height", 540))
    fps = int(cli.get("fps", 30))
    playback_speed = float(cli.get("playback_speed", 4.0))
    output.parent.mkdir(parents=True, exist_ok=True)

    actor = GraspPointGRUActor(checkpoint)
    env = JakaMovingSphereMuJoCo(
        seed=seed,
        belt_speed=bowl_speed,
        grasp_assist=False,
        max_seconds=max_seconds_per_call,
        controller_profile="static_pick",
        pick_place=True,
        task_variant="static_sphere_moving_bowl",
        num_spheres=num_items,
        moving_bowl_queue_spacing=spacing,
        translation_residual_limit=0.03,
        rotation_residual_limit=0.015,
    )
    observation = env.reset(seed)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    if encoder.stdin is None:
        raise RuntimeError("ffmpeg stdin closed unexpectedly")
    frame_period = playback_speed / fps
    next_frame_time = float(env.data.time)
    frames = 0
    calls: list[dict[str, object]] = []

    def write_still(seconds: float) -> None:
        nonlocal frames
        renderer.update_scene(env.data, camera=camera)
        frame = np.ascontiguousarray(renderer.render())
        for _ in range(round(seconds * fps)):
            encoder.stdin.write(frame.tobytes())
            frames += 1

    def write_due_frame() -> None:
        nonlocal next_frame_time, frames
        if float(env.data.time) + 1.0e-9 < next_frame_time:
            return
        renderer.update_scene(env.data, camera=camera)
        encoder.stdin.write(np.ascontiguousarray(renderer.render()).tobytes())
        frames += 1
        next_frame_time += frame_period

    try:
        configure_upstream_camera(camera)
        write_still(1.0)
        configure_supply_camera(camera)
        write_still(1.0)
        configure_workcell_camera(camera)

        for item_index in range(num_items):
            if item_index > 0:
                observation = env.prepare_static_sphere_moving_bowl_call(item_index)
            result = None
            phases: list[str] = []
            action_abs_sum = 0.0
            action_values = 0
            action_max_abs = 0.0
            nonzero_action_steps = 0
            for _ in range(int(max_seconds_per_call / POLICY_DT) + 1):
                network_action = actor.act(observation)
                action_abs_sum += float(np.abs(network_action).sum())
                action_values += int(network_action.size)
                action_max_abs = max(action_max_abs, float(np.abs(network_action).max()))
                nonzero_action_steps += int(np.any(np.abs(network_action) > 1.0e-8))
                result = env.step(network_action)
                observation = result.observation
                phase = str(result.info["option_phase"])
                if not phases or phases[-1] != phase:
                    phases.append(phase)
                write_due_frame()
                if result.done:
                    break
            if result is None or result.outcome != "strict_place_home_success":
                raise RuntimeError(
                    f"pair {item_index + 1} failed: "
                    f"{None if result is None else result.outcome}"
                )
            calls.append(
                {
                    "call": item_index + 1,
                    "outcome": result.outcome,
                    "policy_seconds": round(env.elapsed, 3),
                    "network_control_steps": action_values // 7,
                    "network_nonzero_action_steps": nonzero_action_steps,
                    "network_mean_abs_action": action_abs_sum / max(action_values, 1),
                    "network_max_abs_action": action_max_abs,
                    "phases": phases,
                }
            )

        for _ in range(round(1.0 / env.model.opt.timestep)):
            env._apply_ik()
            env.data.ctrl[env.finger_act] = 0.0
            env._set_pallet_targets(float(env.data.time) + env.model.opt.timestep)
            mujoco.mj_step(env.model, env.data)
            write_due_frame()
        mujoco.mj_forward(env.model, env.data)
        final_pairs = pair_snapshot(env)
        retained = sum(bool(record["retained_and_settled"]) for record in final_pairs)
        if retained != num_items:
            raise RuntimeError(f"only {retained}/{num_items} balls remain in assigned bowls")
        configure_downstream_camera(camera)
        write_still(1.0)
    finally:
        encoder.stdin.close()
        return_code = encoder.wait()
        renderer.close()
        env.close()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    record = {
        "video": str(output),
        "engine": "mujoco",
        "physics_closed_loop": True,
        "robot": "jaka_zu5_ag95",
        "task": "ten_static_spheres_to_ten_visible_moving_bowls",
        "controller": "frozen_checkpoint_residual_plus_geometry_baseline",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "network_called_every_control_step": True,
        "zero_residual_baseline": False,
        "translation_residual_limit": 0.03,
        "rotation_residual_limit": 0.015,
        "policy_calls": num_items,
        "successful_policy_calls": num_items,
        "balls_retained_in_assigned_bowls": num_items,
        "strict_sequence_success": True,
        "bowl_speed_mps": bowl_speed,
        "bowl_spacing_m": spacing,
        "all_spheres_present_at_start": True,
        "all_bowls_present_and_moving_at_start": True,
        "spawned_between_calls": False,
        "world_reset_between_calls": False,
        "grasp_assist": False,
        "actor_speed_input": False,
        "gpu_used": False,
        "opening_cameras": ["wide_upstream_bowl_queue", "static_supply_rack"],
        "closing_camera": "wide_downstream_filled_bowls",
        "playback_speed": playback_speed,
        "video_seconds": round(frames / fps, 3),
        "frames": frames,
        "fps": fps,
        "sphere_outline_added": False,
        "calls": calls,
        "final_pairs": final_pairs,
    }
    output.with_suffix(".json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print("JAKA_TEN_STATIC_TO_TEN_MOVING_VIDEO " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
