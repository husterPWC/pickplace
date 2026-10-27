#!/usr/bin/env python3
"""Render the strict JAKA static-sphere -> 0.10 m/s moving-bowl rollout."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

if os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1":
    raise RuntimeError("native render requires LIBGL_ALWAYS_SOFTWARE=1")
if os.environ.get("MUJOCO_GL") != "glfw":
    raise RuntimeError("native render requires MUJOCO_GL=glfw under Xvfb")
if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
    raise RuntimeError("MuJoCo render must not use a GPU")

import mujoco
import numpy as np

from mujoco_sim.env import POLICY_DT, ROOT
from mujoco_sim.jaka_moving_sphere import JakaMovingSphereMuJoCo
from mujoco_sim.policy_runner import parse_cli


DEFAULT_OUTPUT = ROOT / (
    "artifacts/jaka_zu5_ag95_static_sphere_moving_bowl/mujoco/"
    "speed010_strict_pick_place_home_full_scene_cpu.mp4"
)


def main() -> None:
    cli = parse_cli()
    output = Path(cli.get("out", DEFAULT_OUTPUT)).resolve()
    seed = int(cli.get("seed", 7100))
    bowl_speed = float(cli.get("bowl_speed", 0.10))
    max_seconds = float(cli.get("max_seconds", 12.0))
    width = int(cli.get("width", 960))
    height = int(cli.get("height", 540))
    fps = int(cli.get("fps", 30))
    output.parent.mkdir(parents=True, exist_ok=True)

    env = JakaMovingSphereMuJoCo(
        seed=seed,
        belt_speed=bowl_speed,
        grasp_assist=False,
        max_seconds=max_seconds,
        sphere_start_x=0.45,
        sphere_start_y=0.00,
        controller_profile="static_pick",
        pick_place=True,
        task_variant="static_sphere_moving_bowl",
    )
    env.reset(seed)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.38, 0.32, 0.20)
    camera.distance = 1.55
    camera.azimuth = 132
    camera.elevation = -24

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    result = None
    phases: list[str] = []
    frames = 0
    next_frame_time = 0.0
    try:
        renderer.update_scene(env.data, camera=camera)
        initial = np.ascontiguousarray(renderer.render())
        if encoder.stdin is None:
            raise RuntimeError("ffmpeg stdin closed unexpectedly")
        for _ in range(fps):
            encoder.stdin.write(initial.tobytes())
            frames += 1
        for _ in range(int(max_seconds / POLICY_DT) + 1):
            result = env.step(np.zeros(7, dtype=np.float32))
            phase = str(result.info["option_phase"])
            if not phases or phases[-1] != phase:
                phases.append(phase)
            if env.elapsed + 1.0e-9 >= next_frame_time:
                renderer.update_scene(env.data, camera=camera)
                encoder.stdin.write(np.ascontiguousarray(renderer.render()).tobytes())
                frames += 1
                next_frame_time += 1.0 / fps
            if result.done:
                renderer.update_scene(env.data, camera=camera)
                final = np.ascontiguousarray(renderer.render())
                for _ in range(fps):
                    encoder.stdin.write(final.tobytes())
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
    if result is None or result.outcome != "strict_place_home_success":
        raise RuntimeError(f"refusing failed render: {None if result is None else result.outcome}")

    record = {
        "video": str(output),
        "engine": "mujoco",
        "physics_closed_loop": True,
        "robot": "jaka_zu5_ag95",
        "task": "static_sphere_to_moving_bowl",
        "controller": "geometry_baseline_plus_zero_residual",
        "gpu_used": False,
        "grasp_assist": False,
        "bowl_speed_mps": bowl_speed,
        "outcome": result.outcome,
        "phases": phases,
        "episode_seconds": round(env.elapsed, 3),
        "video_seconds": round(frames / fps, 3),
        "frames": frames,
        "fps": fps,
        "sphere_outline_added": False,
        "final_info": result.info,
    }
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("JAKA_STATIC_TO_MOVING_VIDEO " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
