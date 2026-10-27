#!/usr/bin/env python3
"""Render the complete MuJoCo scene through Mesa llvmpipe (CPU only).

This is a real mesh render, unlike the lightweight diagnostic state plot.
The hard environment checks intentionally prevent accidental NVIDIA/EGL use
on the local machine with the unsafe physical GPU0.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

if os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1":
    raise RuntimeError("native sphere render requires LIBGL_ALWAYS_SOFTWARE=1")
if os.environ.get("MUJOCO_GL") != "glfw":
    raise RuntimeError("native sphere render requires MUJOCO_GL=glfw under the Mesa Xvfb")
if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
    raise RuntimeError("native sphere render requires CUDA_VISIBLE_DEVICES='' or '-1'")

import mujoco
import numpy as np

from mujoco_sim.moving_sphere import (
    BELT_SPEED,
    MAX_SECONDS,
    POLICY_DT,
    ROOT,
    FrankaMovingSphereMuJoCo,
    GraspPointGRUActor,
)
from mujoco_sim.policy_runner import as_bool, parse_cli


DEFAULT_CHECKPOINT = ROOT / (
    "artifacts/moving_sphere_static_bowl/teachers/"
    "moving_sphere_g0_s0d_canonical_coarse_1024_no_rtx/checkpoints/ppo_teacher_262144.pt"
)
DEFAULT_OUTPUT = ROOT / (
    "artifacts/moving_sphere_static_bowl/mujoco/"
    "g0_s0d_262144_mujoco_full_scene_cpu.mp4"
)


def main() -> None:
    cli = parse_cli()
    checkpoint = Path(cli.get("checkpoint", DEFAULT_CHECKPOINT)).resolve()
    output = Path(cli.get("out", DEFAULT_OUTPUT)).resolve()
    seed = int(cli.get("seed", 0))
    belt_speed = float(cli.get("belt_speed", BELT_SPEED))
    width = int(cli.get("width", 960))
    height = int(cli.get("height", 540))
    fps = int(cli.get("fps", 30))
    playback_speed = float(cli.get("playback_speed", 2.0))
    grasp_assist = as_bool(cli.get("grasp_assist", True))
    controller = str(cli.get("controller", "option"))
    intercept_lead_distance = float(cli.get("intercept_lead_distance", 0.08))
    intercept_align_lead_distance = float(cli.get("intercept_align_lead_distance", 0.04))
    pre_descent_raw = cli.get("pre_descent_height", None)
    pre_descent_height = None if pre_descent_raw in (None, "none", "null") else float(pre_descent_raw)
    lift_steps = int(cli.get("lift_steps", 30))
    lift_height = float(cli.get("lift_height", 0.045))
    lift_xy_gain_raw = cli.get("lift_xy_gain", None)
    lift_xy_gain = None if lift_xy_gain_raw in (None, "none", "null") else float(lift_xy_gain_raw)
    max_lift_xy_step_raw = cli.get("max_lift_xy_step", None)
    max_lift_xy_step = (
        None if max_lift_xy_step_raw in (None, "none", "null") else float(max_lift_xy_step_raw)
    )
    preserve_lift_relative_xy = as_bool(cli.get("preserve_lift_relative_xy", False))
    target_xy_feedforward_gain = float(cli.get("target_xy_feedforward_gain", 0.0))
    max_target_xy_feedforward_step_raw = cli.get("max_target_xy_feedforward_step", None)
    max_target_xy_feedforward_step = (
        None
        if max_target_xy_feedforward_step_raw in (None, "none", "null")
        else float(max_target_xy_feedforward_step_raw)
    )
    align_target_xy_feedforward = as_bool(cli.get("align_target_xy_feedforward", False))
    close_requires_raw_alignment = as_bool(cli.get("close_requires_raw_alignment", False))
    max_close_hold_xy_step_raw = cli.get("max_close_hold_xy_step", None)
    max_close_hold_xy_step = (
        None
        if max_close_hold_xy_step_raw in (None, "none", "null")
        else float(max_close_hold_xy_step_raw)
    )
    latch_lift_xy_feedforward = as_bool(cli.get("latch_lift_xy_feedforward", False))
    lift_geometry_relatch_height_raw = cli.get("lift_geometry_relatch_height", None)
    lift_geometry_relatch_height = (
        None
        if lift_geometry_relatch_height_raw in (None, "none", "null")
        else float(lift_geometry_relatch_height_raw)
    )
    sphere_start_x = float(cli.get("sphere_start_x", 0.20))
    belt_upstream_x_raw = cli.get("belt_upstream_x", None)
    belt_upstream_x = (
        None if belt_upstream_x_raw in (None, "none", "null") else float(belt_upstream_x_raw)
    )
    pallet_start_x_raw = cli.get("pallet_start_x", None)
    pallet_start_x = (
        None if pallet_start_x_raw in (None, "none", "null") else float(pallet_start_x_raw)
    )
    lift_timeout_steps = int(cli.get("lift_timeout_steps", 300))
    max_seconds = float(cli.get("max_seconds", MAX_SECONDS))
    pick_place = as_bool(cli.get("pick_place", False))
    bowl_x = float(cli.get("bowl_x", 0.45))
    bowl_y = float(cli.get("bowl_y", 0.0))
    translation_residual_limit = float(
        cli.get("translation_residual_limit", 0.05 if pick_place else 0.10)
    )
    rotation_residual_limit = float(
        cli.get("rotation_residual_limit", 0.025 if pick_place else 0.05)
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    env = FrankaMovingSphereMuJoCo(
        seed=seed,
        belt_speed=belt_speed,
        grasp_assist=grasp_assist,
        controller=controller,
        intercept_lead_distance=intercept_lead_distance,
        intercept_align_lead_distance=intercept_align_lead_distance,
        pre_descent_height=pre_descent_height,
        lift_steps=lift_steps,
        lift_height=lift_height,
        lift_xy_gain=lift_xy_gain,
        max_lift_xy_step=max_lift_xy_step,
        preserve_lift_relative_xy=preserve_lift_relative_xy,
        target_xy_feedforward_gain=target_xy_feedforward_gain,
        max_target_xy_feedforward_step=max_target_xy_feedforward_step,
        align_target_xy_feedforward=align_target_xy_feedforward,
        close_requires_raw_alignment=close_requires_raw_alignment,
        max_close_hold_xy_step=max_close_hold_xy_step,
        latch_lift_xy_feedforward=latch_lift_xy_feedforward,
        lift_geometry_relatch_height=lift_geometry_relatch_height,
        sphere_start_x=sphere_start_x,
        belt_upstream_x=belt_upstream_x,
        pallet_start_x=pallet_start_x,
        lift_timeout_steps=lift_timeout_steps,
        max_seconds=max_seconds,
        pick_place=pick_place,
        destination_bowl_pos=(bowl_x, bowl_y, 0.0),
        translation_residual_limit=translation_residual_limit,
        rotation_residual_limit=rotation_residual_limit,
    )
    actor = GraspPointGRUActor(checkpoint)
    observation = env.reset(seed)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.35, 0.08, 0.25)
    camera.distance = 1.55
    camera.azimuth = 132
    camera.elevation = -23

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    next_frame_time = 0.0
    frame_period = playback_speed / fps
    frames = 0
    result = None
    try:
        # Hold the complete reset scene briefly so every object is visible.
        renderer.update_scene(env.data, camera=camera)
        reset_frame = np.ascontiguousarray(renderer.render())
        if encoder.stdin is None:
            raise RuntimeError("ffmpeg stdin closed unexpectedly")
        for _ in range(fps):
            encoder.stdin.write(reset_frame.tobytes())
            frames += 1

        for _ in range(int(max_seconds / POLICY_DT) + 1):
            result = env.step(actor.act(observation))
            observation = result.observation
            if env.elapsed + 1.0e-9 >= next_frame_time:
                renderer.update_scene(env.data, camera=camera)
                frame = np.ascontiguousarray(renderer.render())
                encoder.stdin.write(frame.tobytes())
                frames += 1
                next_frame_time += frame_period
            if result.done:
                renderer.update_scene(env.data, camera=camera)
                final_frame = np.ascontiguousarray(renderer.render())
                for _ in range(fps):
                    encoder.stdin.write(final_frame.tobytes())
                    frames += 1
                break
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
        renderer.close()
        env.close()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    if result is None:
        raise RuntimeError("rollout produced no steps")

    record = {
        "video": str(output),
        "checkpoint": str(checkpoint),
        "engine": "mujoco",
        "physics_closed_loop": True,
        "visualizer": "mujoco_native_mesh_mesa_llvmpipe",
        "gpu_used": False,
        "policy_input_has_explicit_velocity": False,
        "policy_input_has_contact_or_force": False,
        "controller": controller,
        "intercept_lead_distance": intercept_lead_distance,
        "intercept_align_lead_distance": intercept_align_lead_distance,
        "pre_descent_height": pre_descent_height,
        "lift_steps": lift_steps,
        "lift_height": lift_height,
        "lift_xy_gain": lift_xy_gain,
        "max_lift_xy_step": max_lift_xy_step,
        "preserve_lift_relative_xy": preserve_lift_relative_xy,
        "target_xy_feedforward_gain": target_xy_feedforward_gain,
        "max_target_xy_feedforward_step": max_target_xy_feedforward_step,
        "align_target_xy_feedforward": align_target_xy_feedforward,
        "close_requires_raw_alignment": close_requires_raw_alignment,
        "max_close_hold_xy_step": max_close_hold_xy_step,
        "latch_lift_xy_feedforward": latch_lift_xy_feedforward,
        "lift_geometry_relatch_height": lift_geometry_relatch_height,
        "sphere_start_x": sphere_start_x,
        "belt_upstream_x": belt_upstream_x,
        "pallet_start_x": pallet_start_x,
        "lift_timeout_steps": lift_timeout_steps,
        "max_seconds": max_seconds,
        "pick_place": pick_place,
        "destination_bowl_pos": [bowl_x, bowl_y, 0.0],
        "translation_residual_limit": translation_residual_limit,
        "rotation_residual_limit": rotation_residual_limit,
        "grasp_assist": grasp_assist,
        "seed": seed,
        "belt_speed": belt_speed,
        "outcome": result.outcome,
        "episode_seconds": round(env.elapsed, 3),
        "video_seconds": round(frames / fps, 3),
        "playback_speed": playback_speed,
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "min_translation": env.diagnostics.min_translation,
        "min_rotation": env.diagnostics.min_rotation,
        "max_lift": env.diagnostics.max_lift,
        "option_triggered": env.diagnostics.option_triggered,
        "final_info": result.info,
    }
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    print("MUJOCO_NATIVE_SPHERE_VIDEO " + json.dumps(record), flush=True)
    print(f"MUJOCO_NATIVE_SPHERE_VIDEO_DONE {output}", flush=True)


if __name__ == "__main__":
    main()
