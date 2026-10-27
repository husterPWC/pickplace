#!/usr/bin/env python3
"""Render a true closed-loop JAKA ZU5+AG95 MuJoCo rollout with Mesa."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

if os.environ.get("LIBGL_ALWAYS_SOFTWARE") != "1":
    raise RuntimeError("JAKA native render requires LIBGL_ALWAYS_SOFTWARE=1")
if os.environ.get("MUJOCO_GL") != "glfw":
    raise RuntimeError("JAKA native render requires MUJOCO_GL=glfw under Xvfb")
if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
    raise RuntimeError("JAKA native render must not use a GPU")

import mujoco
import numpy as np

from mujoco_sim.env import POLICY_DT, ROOT
from mujoco_sim.jaka_moving_sphere import DEFAULT_CHECKPOINT, JakaMovingSphereMuJoCo
from mujoco_sim.moving_sphere import GraspPointGRUActor
from mujoco_sim.policy_runner import as_bool, parse_cli


DEFAULT_OUTPUT = ROOT / (
    "artifacts/jaka_zu5_ag95_sphere/mujoco/"
    "j0_s1_speed004_pure_contact_full_scene_cpu.mp4"
)


def main() -> None:
    cli = parse_cli()
    checkpoint = Path(cli.get("checkpoint", DEFAULT_CHECKPOINT)).resolve()
    output = Path(cli.get("out", DEFAULT_OUTPUT)).resolve()
    seed = int(cli.get("seed", 31))
    belt_speed = float(cli.get("belt_speed", 0.04))
    max_seconds = float(cli.get("max_seconds", 45.0))
    grasp_assist = as_bool(cli.get("grasp_assist", False))
    controller_profile = str(cli.get("controller_profile", "j0_s1"))
    pick_place = as_bool(cli.get("pick_place", controller_profile == "j1"))
    bowl_pos = (
        float(cli.get("bowl_x", 0.45)),
        float(cli.get("bowl_y", 0.0)),
        float(cli.get("bowl_z", 0.0)),
    )
    width = int(cli.get("width", 960))
    height = int(cli.get("height", 540))
    fps = int(cli.get("fps", 30))
    playback_speed = float(cli.get("playback_speed", 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)

    env = JakaMovingSphereMuJoCo(
        seed=seed,
        belt_speed=belt_speed,
        grasp_assist=grasp_assist,
        max_seconds=max_seconds,
        controller_profile=controller_profile,
        pick_place=pick_place,
        destination_bowl_pos=bowl_pos,
    )
    actor = GraspPointGRUActor(checkpoint)
    observation = env.reset(seed)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.35, 0.10, 0.25)
    camera.distance = 1.35
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
    frame_period = playback_speed / fps
    next_frame_time = 0.0
    frames = 0
    result = None
    phases: list[str] = []
    try:
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
            phase = str(result.info["option_phase"])
            if not phases or phases[-1] != phase:
                phases.append(phase)
            if env.elapsed + 1.0e-9 >= next_frame_time:
                renderer.update_scene(env.data, camera=camera)
                encoder.stdin.write(np.ascontiguousarray(renderer.render()).tobytes())
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
    expected_outcome = "strict_place_success" if pick_place else "strict_lift_success"
    if result.outcome != expected_outcome:
        raise RuntimeError(f"refusing to accept failed JAKA MuJoCo render: {result.outcome}")

    record = {
        "video": str(output),
        "checkpoint": str(checkpoint),
        "engine": "mujoco",
        "physics_closed_loop": True,
        "robot": "jaka_zu5_ag95",
        "visualizer": "mujoco_native_mesh_mesa_llvmpipe",
        "gpu_used": False,
        "isaac_trajectory_replay": False,
        "grasp_assist": grasp_assist,
        "controller_profile": controller_profile,
        "pick_place": pick_place,
        "destination_bowl_pos": list(bowl_pos),
        "seed": seed,
        "belt_speed": belt_speed,
        "outcome": result.outcome,
        "phases": phases,
        "episode_seconds": round(env.elapsed, 3),
        "video_seconds": round(frames / fps, 3),
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "sphere_outline_added": False,
        "final_info": result.info,
    }
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("JAKA_MUJOCO_NATIVE_VIDEO " + json.dumps(record), flush=True)
    print(f"JAKA_MUJOCO_NATIVE_VIDEO_DONE {output}", flush=True)


if __name__ == "__main__":
    main()
