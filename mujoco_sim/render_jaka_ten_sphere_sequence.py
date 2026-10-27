#!/usr/bin/env python3
"""Render ten calls to the frozen JAKA single-ball policy in one world."""

from __future__ import annotations

import json
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
from mujoco_sim.evaluate_jaka_ten_sphere_sequence import _sphere_snapshot
from mujoco_sim.jaka_moving_sphere import DEFAULT_CHECKPOINT, JakaMovingSphereMuJoCo
from mujoco_sim.moving_sphere import GraspPointGRUActor
from mujoco_sim.policy_runner import parse_cli


DEFAULT_OUTPUT = ROOT / (
    "artifacts/jaka_zu5_ag95_sphere/mujoco/"
    "ten_sphere_repeated_policy_speed008_full_scene_cpu.mp4"
)


def main() -> None:
    cli = parse_cli()
    checkpoint = Path(cli.get("checkpoint", DEFAULT_CHECKPOINT)).resolve()
    output = Path(cli.get("out", DEFAULT_OUTPUT)).resolve()
    seed = int(cli.get("seed", 8200))
    num_spheres = int(cli.get("num_spheres", 10))
    belt_speed = float(cli.get("belt_speed", 0.08))
    max_seconds_per_call = float(cli.get("max_seconds_per_call", 12.0))
    width = int(cli.get("width", 960))
    height = int(cli.get("height", 540))
    fps = int(cli.get("fps", 30))
    playback_speed = float(cli.get("playback_speed", 4.0))
    output.parent.mkdir(parents=True, exist_ok=True)

    actor = GraspPointGRUActor(checkpoint)
    env = JakaMovingSphereMuJoCo(
        seed=seed,
        belt_speed=belt_speed,
        grasp_assist=False,
        max_seconds=max_seconds_per_call,
        controller_profile="j1",
        pick_place=True,
        destination_bowl_pos=(0.45, 0.0, 0.0),
        num_spheres=num_spheres,
    )
    observation = env.reset(seed)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.36, 0.12, 0.22)
    camera.distance = 1.40
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
    if encoder.stdin is None:
        raise RuntimeError("ffmpeg stdin closed unexpectedly")
    frame_period = playback_speed / fps
    next_frame_time = float(env.data.time)
    frames = 0
    calls: list[dict[str, object]] = []

    def write_due_frame() -> None:
        nonlocal next_frame_time, frames
        if float(env.data.time) + 1.0e-9 < next_frame_time:
            return
        renderer.update_scene(env.data, camera=camera)
        encoder.stdin.write(np.ascontiguousarray(renderer.render()).tobytes())
        frames += 1
        next_frame_time += frame_period

    try:
        renderer.update_scene(env.data, camera=camera)
        initial = np.ascontiguousarray(renderer.render())
        for _ in range(fps):
            encoder.stdin.write(initial.tobytes())
            frames += 1
        for sphere_index in range(num_spheres):
            if sphere_index > 0:
                observation = env.prepare_sphere_call(sphere_index)
            result = None
            phases: list[str] = []
            for _ in range(int(max_seconds_per_call / POLICY_DT) + 1):
                result = env.step(actor.act(observation))
                observation = result.observation
                phase = str(result.info["option_phase"])
                if not phases or phases[-1] != phase:
                    phases.append(phase)
                write_due_frame()
                if result.done:
                    break
            if result is None or result.outcome != "strict_place_success":
                raise RuntimeError(
                    f"sphere {sphere_index + 1} failed: "
                    f"{None if result is None else result.outcome}"
                )
            return_result = env.return_to_intercept_ready(on_step=write_due_frame)
            if not bool(return_result["success"]):
                raise RuntimeError(f"sphere {sphere_index + 1} failed to return ready")
            calls.append(
                {
                    "call": sphere_index + 1,
                    "outcome": result.outcome,
                    "policy_seconds": round(env.elapsed, 3),
                    "phases": phases,
                    "return_to_ready": return_result,
                }
            )
        for _ in range(round(1.0 / env.model.opt.timestep)):
            env._apply_ik()
            env.data.ctrl[env.finger_act] = 0.0
            mujoco.mj_step(env.model, env.data)
            write_due_frame()
        mujoco.mj_forward(env.model, env.data)
        final_spheres = _sphere_snapshot(env)
        retained = sum(bool(record["retained_and_settled"]) for record in final_spheres)
        if retained != num_spheres:
            raise RuntimeError(f"only {retained}/{num_spheres} balls remain settled in bowl")
        renderer.update_scene(env.data, camera=camera)
        final = np.ascontiguousarray(renderer.render())
        for _ in range(fps):
            encoder.stdin.write(final.tobytes())
            frames += 1
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
        "checkpoint": str(checkpoint),
        "same_actor_reused": True,
        "policy_calls": num_spheres,
        "successful_policy_calls": num_spheres,
        "balls_retained_and_settled": num_spheres,
        "strict_sequence_success": True,
        "belt_speed_mps": belt_speed,
        "grasp_assist": False,
        "world_reset_between_calls": False,
        "gpu_used": False,
        "playback_speed": playback_speed,
        "video_seconds": round(frames / fps, 3),
        "frames": frames,
        "fps": fps,
        "sphere_outline_added": False,
        "calls": calls,
        "final_spheres": final_spheres,
    }
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("JAKA_TEN_SPHERE_VIDEO " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
