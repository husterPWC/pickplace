#!/usr/bin/env python3
"""Render one closed-loop MuJoCo policy episode to an H.264 MP4."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

from mujoco_sim.env import FrankaPickPlaceMuJoCo, POLICY_DT, ROOT
from mujoco_sim.policy_runner import as_bool, load_actor, parse_cli


DEFAULT_CHECKPOINT = ROOT / "artifacts/mujoco/adaptation/franka_v3c_mujoco_adapted.pt"
DEFAULT_OUTPUT = ROOT / "artifacts/mujoco/franka_v3c_adapted_closed_loop.mp4"


def main() -> None:
    cli = parse_cli()
    checkpoint = Path(cli.get("checkpoint", DEFAULT_CHECKPOINT)).resolve()
    output = Path(cli.get("out", DEFAULT_OUTPUT)).resolve()
    seed = int(cli.get("seed", 0))
    randomize = as_bool(cli.get("randomize", False))
    belt_speed = float(cli.get("belt_speed", 0.08))
    width = int(cli.get("width", 1280))
    height = int(cli.get("height", 720))
    fps = int(cli.get("fps", round(1.0 / POLICY_DT)))
    start_hold = int(float(cli.get("start_hold", 1.0)) * fps)
    end_hold = int(float(cli.get("end_hold", 2.0)) * fps)

    output.parent.mkdir(parents=True, exist_ok=True)
    env = FrankaPickPlaceMuJoCo(
        seed=seed,
        randomize=randomize,
        grasp_assist=True,
        belt_speed=belt_speed,
    )
    actor = load_actor(checkpoint)
    actor.reset()
    observation = env.reset(seed)

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.40, 0.05, 0.22)
    camera.distance = 1.42
    camera.azimuth = 132
    camera.elevation = -24

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)

    def render_frame() -> np.ndarray:
        renderer.update_scene(env.data, camera=camera)
        return np.ascontiguousarray(renderer.render())

    frames = 0
    result = None
    try:
        first = render_frame()
        for _ in range(start_hold):
            encoder.stdin.write(first.tobytes())
            frames += 1
        for _ in range(600):
            result = env.step(actor.act(observation))
            observation = result.observation
            frame = render_frame()
            encoder.stdin.write(frame.tobytes())
            frames += 1
            if result.done:
                for _ in range(end_hold):
                    encoder.stdin.write(frame.tobytes())
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
    if result is None or not result.done:
        raise RuntimeError("policy episode did not terminate before the render step limit")
    if result.outcome != "success":
        raise RuntimeError(f"refusing to accept a failed render: {result.outcome}")

    sidecar = output.with_suffix(".json")
    record = {
        "video": str(output),
        "checkpoint": str(checkpoint),
        "seed": seed,
        "randomize": randomize,
        "belt_speed": belt_speed,
        "outcome": result.outcome,
        "episode_seconds": round(env.elapsed, 3),
        "video_seconds": round(frames / fps, 3),
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "final_info": result.info,
    }
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    print("MUJOCO_VIDEO " + json.dumps(record), flush=True)
    print(f"MUJOCO_VIDEO_DONE {output}", flush=True)


if __name__ == "__main__":
    main()
