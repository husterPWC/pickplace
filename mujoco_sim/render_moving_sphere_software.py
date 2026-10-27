#!/usr/bin/env python3
"""CPU-only video of a true closed-loop moving-sphere MuJoCo rollout.

MuJoCo still advances every physical state and the policy is queried at every
control step.  Pillow merely visualizes those MuJoCo states, avoiding EGL and
therefore avoiding both physical GPUs on machines with a fragile display
driver.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
DEFAULT_OUTPUT = ROOT / "artifacts/moving_sphere_static_bowl/mujoco/g0_s0d_262144_closed_loop_cpu.mp4"


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


class StateRenderer:
    def __init__(self, env: FrankaMovingSphereMuJoCo, width: int, height: int, label: str):
        self.env = env
        self.width = width
        self.height = height
        self.label = label
        self.title_font = _font(28)
        self.text_font = _font(20)
        self.small_font = _font(16)
        self.arm_bodies = [env.model.body(f"link{i}").id for i in range(8)] + [env.model.body("hand").id]
        self.tcp_tail: list[np.ndarray] = []
        self.sphere_tail: list[np.ndarray] = []

    @staticmethod
    def _map(value: float, low: float, high: float, start: int, end: int) -> int:
        return int(start + (value - low) / (high - low) * (end - start))

    def _top(self, xyz: np.ndarray) -> tuple[int, int]:
        x = self._map(float(xyz[0]), -0.15, 0.90, 65, self.width // 2 - 45)
        y = self._map(float(xyz[1]), -0.40, 0.55, self.height - 55, 125)
        return x, y

    def _side(self, xyz: np.ndarray) -> tuple[int, int]:
        x = self._map(float(xyz[0]), -0.15, 0.90, self.width // 2 + 45, self.width - 65)
        z = self._map(float(xyz[2]), -0.04, 0.90, self.height - 55, 125)
        return x, z

    @staticmethod
    def _polyline(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str, width: int) -> None:
        if len(points) > 1:
            draw.line(points, fill="#11151a", width=width + 6, joint="curve")
            draw.line(points, fill=color, width=width, joint="curve")

    def _arm(self, draw: ImageDraw.ImageDraw, project) -> None:
        xyz = [self.env.data.xpos[body].copy() for body in self.arm_bodies]
        points = [project(p) for p in xyz]
        self._polyline(draw, points, "#e4e8ec", 18)
        for point in points:
            draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), fill="#303943", outline="#f3f5f7", width=2)
        tcp = project(self.env.data.site_xpos[self.env.tcp_site])
        draw.ellipse((tcp[0] - 9, tcp[1] - 9, tcp[0] + 9, tcp[1] + 9), fill="#ff7b35", outline="white", width=2)

    def frame(self, info: dict[str, float | bool | str], outcome: str | None) -> np.ndarray:
        image = Image.new("RGB", (self.width, self.height), "#131820")
        draw = ImageDraw.Draw(image)
        split = self.width // 2
        draw.rectangle((0, 0, self.width, 96), fill="#202936")
        draw.line((split, 105, split, self.height - 30), fill="#66717f", width=2)
        draw.text((30, 18), "MuJoCo closed-loop sim-to-sim", font=self.title_font, fill="#f4f7fb")
        draw.text((30, 57), self.label, font=self.small_font, fill="#9fd2ff")
        draw.text((split + 25, 28), f"t={self.env.elapsed:5.2f}s   phase={info['option_phase']}", font=self.text_font, fill="#f4f7fb")
        draw.text((split + 25, 60), f"min TCP error={self.env.diagnostics.min_translation:.3f} m   lift={self.env.diagnostics.max_lift:.3f} m", font=self.small_font, fill="#f4c56d")

        # Top-view workcell.
        draw.text((30, 104), "TOP VIEW (x-y)", font=self.small_font, fill="#b9c3ce")
        belt_a = self._top(np.asarray([-0.10, 0.15, 0]))
        belt_b = self._top(np.asarray([0.85, 0.45, 0]))
        draw.rectangle((belt_a[0], belt_b[1], belt_b[0], belt_a[1]), fill="#262d33", outline="#697581", width=3)
        # Conveyor motion direction.
        arrow_y = self._top(np.asarray([0.0, 0.49, 0]))[1]
        draw.line((self._top(np.asarray([0.05, 0, 0]))[0], arrow_y, self._top(np.asarray([0.75, 0, 0]))[0], arrow_y), fill="#77aee8", width=4)
        draw.polygon([(self._top(np.asarray([0.75, 0, 0]))[0], arrow_y), (self._top(np.asarray([0.70, 0, 0]))[0], arrow_y - 9), (self._top(np.asarray([0.70, 0, 0]))[0], arrow_y + 9)], fill="#77aee8")
        # Static bowl at the training destination.
        bowl = self._top(np.asarray([0.45, -0.10, 0]))
        draw.ellipse((bowl[0] - 42, bowl[1] - 42, bowl[0] + 42, bowl[1] + 42), fill="#267fa6", outline="#8cddff", width=5)
        draw.ellipse((bowl[0] - 28, bowl[1] - 28, bowl[0] + 28, bowl[1] + 28), fill="#15232c", outline="#68c5e7", width=3)
        pallet_xyz = self.env.data.xpos[self.env.model.body("pallet").id]
        pallet = self._top(pallet_xyz)
        draw.rounded_rectangle((pallet[0] - 48, pallet[1] - 55, pallet[0] + 48, pallet[1] + 55), radius=8, fill="#555f68", outline="#aab3bb", width=3)

        sphere = self.env.data.xpos[self.env.food_body].copy()
        tcp = self.env.data.site_xpos[self.env.tcp_site].copy()
        self.sphere_tail.append(sphere.copy())
        self.tcp_tail.append(tcp.copy())
        self.sphere_tail = self.sphere_tail[-100:]
        self.tcp_tail = self.tcp_tail[-100:]
        if len(self.sphere_tail) > 1:
            draw.line([self._top(p) for p in self.sphere_tail], fill="#52d77e", width=3)
            draw.line([self._top(p) for p in self.tcp_tail], fill="#ff9b62", width=3)
        self._arm(draw, self._top)
        ball = self._top(sphere)
        draw.ellipse((ball[0] - 13, ball[1] - 13, ball[0] + 13, ball[1] + 13), fill="#43e36f", outline="#eaffef", width=3)

        # Side-view workcell.
        draw.text((split + 30, 104), "SIDE VIEW (x-z)", font=self.small_font, fill="#b9c3ce")
        table_a = self._side(np.asarray([-0.10, 0, 0]))
        table_b = self._side(np.asarray([0.85, 0, -0.04]))
        draw.rectangle((table_a[0], table_a[1], table_b[0], table_b[1]), fill="#3d4249", outline="#7c858f", width=2)
        belt_a_s = self._side(np.asarray([-0.10, 0, 0.02]))
        belt_b_s = self._side(np.asarray([0.85, 0, 0.0]))
        draw.rectangle((belt_a_s[0], belt_a_s[1], belt_b_s[0], belt_b_s[1]), fill="#252c32", outline="#798591", width=3)
        bowl_s = self._side(np.asarray([0.45, 0, 0.0]))
        draw.arc((bowl_s[0] - 42, bowl_s[1] - 42, bowl_s[0] + 42, bowl_s[1] + 22), 0, 180, fill="#79d2f1", width=7)
        self._arm(draw, self._side)
        if len(self.sphere_tail) > 1:
            draw.line([self._side(p) for p in self.sphere_tail], fill="#52d77e", width=3)
            draw.line([self._side(p) for p in self.tcp_tail], fill="#ff9b62", width=3)
        ball_s = self._side(sphere)
        draw.ellipse((ball_s[0] - 13, ball_s[1] - 13, ball_s[0] + 13, ball_s[1] + 13), fill="#43e36f", outline="#eaffef", width=3)

        status = outcome or "running"
        status_color = "#61e68b" if outcome == "strict_lift_success" else ("#ff8a72" if outcome else "#9fd2ff")
        draw.text((30, self.height - 32), f"actual MuJoCo state | belt={self.env.belt_speed:.3f} m/s | {status}", font=self.small_font, fill=status_color)
        return np.asarray(image)


def main() -> None:
    cli = parse_cli()
    checkpoint = Path(cli.get("checkpoint", DEFAULT_CHECKPOINT)).resolve()
    output = Path(cli.get("out", DEFAULT_OUTPUT)).resolve()
    seed = int(cli.get("seed", 0))
    belt_speed = float(cli.get("belt_speed", BELT_SPEED))
    width = int(cli.get("width", 1280))
    height = int(cli.get("height", 720))
    fps = int(cli.get("fps", 30))
    playback_speed = float(cli.get("playback_speed", 2.0))
    grasp_assist = as_bool(cli.get("grasp_assist", True))
    label = cli.get("label", checkpoint.stem)
    output.parent.mkdir(parents=True, exist_ok=True)

    env = FrankaMovingSphereMuJoCo(seed=seed, belt_speed=belt_speed, grasp_assist=grasp_assist)
    actor = GraspPointGRUActor(checkpoint)
    observation = env.reset(seed)
    renderer = StateRenderer(env, width, height, label)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    next_frame_time = 0.0
    frame_period = playback_speed / fps
    frames = 0
    result = None
    try:
        for _ in range(int(MAX_SECONDS / POLICY_DT) + 1):
            result = env.step(actor.act(observation))
            observation = result.observation
            if env.elapsed + 1e-9 >= next_frame_time:
                frame = renderer.frame(result.info, result.outcome)
                if encoder.stdin is None:
                    raise RuntimeError("ffmpeg stdin closed unexpectedly")
                encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
                frames += 1
                next_frame_time += frame_period
            if result.done:
                # Hold the final diagnostic frame for one output second.
                frame = renderer.frame(result.info, result.outcome)
                if encoder.stdin is not None:
                    for _ in range(fps):
                        encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
                        frames += 1
                break
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
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
        "visualizer": "cpu_state_renderer",
        "gpu_used": False,
        "policy_input_has_explicit_velocity": False,
        "policy_input_has_contact_or_force": False,
        "seed": seed,
        "belt_speed": belt_speed,
        "grasp_assist": grasp_assist,
        "outcome": result.outcome,
        "episode_seconds": round(env.elapsed, 3),
        "video_seconds": round(frames / fps, 3),
        "playback_speed": playback_speed,
        "frames": frames,
        "fps": fps,
        "min_translation": env.diagnostics.min_translation,
        "min_rotation": env.diagnostics.min_rotation,
        "max_lift": env.diagnostics.max_lift,
        "option_triggered": env.diagnostics.option_triggered,
        "final_info": result.info,
    }
    sidecar = output.with_suffix(".json")
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    print("MUJOCO_SPHERE_CPU_VIDEO " + json.dumps(record), flush=True)
    print(f"MUJOCO_SPHERE_CPU_VIDEO_DONE {output}", flush=True)


if __name__ == "__main__":
    main()
