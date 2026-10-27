#!/usr/bin/env python3
"""Render a recorded Franka teacher rollout without Isaac RTX.

The input is produced by ``pipeline/0_state_teacher/demo.py trace=<file.npz>``.  Robot link poses are
measured by PhysX; this script only transforms the bundled Franka STL meshes and draws simple cell geometry.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


def options() -> dict[str, str]:
    return dict(arg.split("=", 1) for arg in sys.argv[1:] if "=" in arg)


OPTS = options()
TRACE = Path(OPTS.get("trace", ROOT / "artifacts/videos/franka_v3c/v3c_best_trace.npz"))
VIDEO = Path(OPTS.get("out", TRACE.with_suffix(".mp4")))
POSTER = VIDEO.with_name(VIDEO.stem + "_preview.png")
MESH_DIR = (
    ROOT
    / ".venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.robot_motion.pink"
    / "robot_configurations/franka/meshes"
)


def box(center, size) -> np.ndarray:
    c, h = np.asarray(center, float), 0.5 * np.asarray(size, float)
    v = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * h + c
    f = np.asarray(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        dtype=int,
    )
    return v[f]


def sphere(center, radius=0.025) -> np.ndarray:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    return np.asarray(mesh.vertices)[np.asarray(mesh.faces)] + np.asarray(center)


def ring(center, radius=0.075, height=0.045) -> np.ndarray:
    """Open-looking destination bowl: short outer wall plus a thin bottom disk."""
    wall = trimesh.creation.annulus(r_min=radius * 0.82, r_max=radius, height=height)
    wall.apply_translation(np.asarray(center) + [0, 0, height / 2])
    bottom = trimesh.creation.cylinder(radius=radius, height=0.008, sections=24)
    bottom.apply_translation(np.asarray(center) + [0, 0, 0.004])
    meshes = [wall, bottom]
    return np.concatenate([np.asarray(m.vertices)[np.asarray(m.faces)] for m in meshes])


def shaded(color: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    light = np.asarray([-0.45, -0.35, 0.82])
    light /= np.linalg.norm(light)
    intensity = 0.42 + 0.58 * np.abs(normal @ light)
    return np.clip(color[None] * intensity[:, None] + 0.06, 0, 1)


def load_robot(body_names: list[str]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    colors = {
        "panda_link0": np.asarray([0.22, 0.25, 0.29]),
        "panda_link1": np.asarray([0.93, 0.93, 0.91]),
        "panda_link2": np.asarray([0.22, 0.25, 0.29]),
        "panda_link3": np.asarray([0.93, 0.93, 0.91]),
        "panda_link4": np.asarray([0.22, 0.25, 0.29]),
        "panda_link5": np.asarray([0.93, 0.93, 0.91]),
        "panda_link6": np.asarray([0.22, 0.25, 0.29]),
        "panda_link7": np.asarray([0.93, 0.93, 0.91]),
        "panda_hand": np.asarray([0.12, 0.14, 0.17]),
        "panda_leftfinger": np.asarray([0.08, 0.09, 0.10]),
        "panda_rightfinger": np.asarray([0.08, 0.09, 0.10]),
    }
    result = []
    for name, color in colors.items():
        if name not in body_names:
            continue
        path = MESH_DIR / f"{name}.stl"
        mesh = trimesh.load_mesh(path, process=False)
        if len(mesh.faces) > 700:
            points, _ = trimesh.sample.sample_surface(mesh, 500)
            mesh = trimesh.Trimesh(vertices=points, process=False).convex_hull
        triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        result.append((body_names.index(name), triangles, color))
    if len(result) < 9:
        raise RuntimeError(f"Only matched {len(result)} Franka bodies; names={body_names}")
    return result


def main() -> None:
    data = np.load(TRACE)
    body_names = [str(x) for x in data["body_names"]]
    robot = load_robot(body_names)
    n_frames = len(data["time"])
    if "seconds" in OPTS:
        n_frames = min(n_frames, int(np.searchsorted(data["time"], float(OPTS["seconds"]), side="right")))
    fps = max(1, round(1.0 / float(data["dt"])))
    events = __import__("json").loads(str(data["events"]))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#111820")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#111820")
    # Look from the conveyor side: bowls/food stay in front instead of being hidden behind the table/arm.
    ax.view_init(elev=25, azim=121)
    ax.set_proj_type("persp", focal_length=0.85)
    ax.set_xlim(-0.28, 1.08)
    ax.set_ylim(-0.55, 0.57)
    ax.set_zlim(-0.03, 1.02)
    ax.set_box_aspect((1.36, 1.12, 1.05))
    ax.set_xlabel("X / m", color="#d9e0e6")
    ax.set_ylabel("Y / m", color="#d9e0e6")
    ax.set_zlabel("Z / m", color="#d9e0e6")
    ax.tick_params(colors="#aab5bf")
    ax.grid(True, color="#394754", alpha=0.35)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.10, 0.13, 0.16, 1.0))

    # The scene is deliberately simple; the moving robot geometry and poses are the real simulated ones.
    static = [
        (box([0.45, 0, -0.04], [1.50, 1.10, 0.08]), np.asarray([0.25, 0.27, 0.30]), 0.16),
        (box([0.45, 0.30, 0.012], [1.10, 0.30, 0.024]), np.asarray([0.055, 0.065, 0.075]), 0.70),
        (ring([0.45, -0.10, 0.0], 0.11, 0.025), np.asarray([0.20, 0.50, 0.66]), 0.95),
    ]
    for triangles, color, alpha in static:
        poly = Poly3DCollection(triangles, linewidths=0.05, edgecolors=(0, 0, 0, 0.15))
        poly.set_facecolors(shaded(color, triangles))
        poly.set_alpha(alpha)
        ax.add_collection3d(poly)

    robot_collections = []
    for body_id, triangles, color in robot:
        poly = Poly3DCollection([], linewidths=0.03, edgecolors=(0, 0, 0, 0.10))
        ax.add_collection3d(poly)
        robot_collections.append((poly, body_id, triangles, color))
    bowl_collections = []
    for _ in range(data["bowl_pos"].shape[1]):
        poly = Poly3DCollection([], linewidths=0.08, edgecolors=(0.05, 0.05, 0.05, 0.35))
        ax.add_collection3d(poly)
        bowl_collections.append(poly)
    food_collections = []
    for _ in range(data["food_pos"].shape[1]):
        poly = Poly3DCollection([], linewidths=0.05, edgecolors=(0, 0, 0, 0.25))
        ax.add_collection3d(poly)
        food_collections.append(poly)

    title = fig.text(0.055, 0.945, "Franka v3c teacher | Isaac PhysX policy rollout", color="white", fontsize=17, weight="bold")
    status = fig.text(0.055, 0.895, "", color="#e4ecf3", fontsize=12)
    detail = fig.text(0.055, 0.855, "", color="#aebac4", fontsize=10)
    fig.legend(
        handles=[Patch(color="#d27835", label="active food"), Patch(color="#3895c5", label="moving bowls"),
                 Patch(color="#efefec", label="Franka mesh")],
        loc="upper right", bbox_to_anchor=(0.955, 0.955), frameon=False, labelcolor="white", ncol=3,
    )

    def draw(frame: int):
        pos, quat = data["body_pos"][frame], data["body_quat"][frame]
        artists = []
        for poly, body_id, local_tri, color in robot_collections:
            tri = Rotation.from_quat(quat[body_id]).apply(local_tri.reshape(-1, 3)).reshape(-1, 3, 3)
            tri += pos[body_id]
            poly.set_verts(tri)
            poly.set_facecolors(shaded(color, tri))
            artists.append(poly)
        for i, poly in enumerate(bowl_collections):
            tri = ring(data["bowl_pos"][frame, i], 0.075, 0.045)
            poly.set_verts(tri)
            poly.set_facecolors(shaded(np.asarray([0.15, 0.50, 0.72]), tri))
            artists.append(poly)
        active = str(data["active_food"][frame])
        for i, poly in enumerate(food_collections):
            p = data["food_pos"][frame, i]
            # Spare objects are parked below the floor and do not clutter the view.
            if p[2] < -0.04:
                poly.set_verts([])
                continue
            tri = sphere(p)
            color = np.asarray([0.90, 0.38, 0.12]) if str(data["food_names"][i]) == active else np.asarray([0.46, 0.25, 0.12])
            poly.set_verts(tri)
            poly.set_facecolors(shaded(color, tri))
            artists.append(poly)
        t = float(data["time"][frame])
        placed = int(data["placed"][frame])
        home_d = float(data["tcp_home_distance"][frame])
        target = int(data["target_bowl"][frame])
        recent = [e["event"] for e in events if 0 <= t - float(e["t"]) < 0.8]
        status.set_text(f"t={t:5.2f} s    placed={placed}    target bowl={target + 1 if target >= 0 else '-'}    TCP→Home={home_d:.3f} m")
        detail.set_text(("event: " + ", ".join(recent)) if recent else "deterministic policy; no scripted Home return")
        return artists + [title, status, detail]

    # A placement frame is more useful than the midpoint as the preview.
    placed = data["placed"]
    changed = np.flatnonzero(np.diff(placed, prepend=placed[0]))
    preview = int(changed[0]) if len(changed) else n_frames // 2
    draw(preview)
    POSTER.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(POSTER, dpi=100, facecolor=fig.get_facecolor())
    if OPTS.get("preview_only", "false").lower() == "true":
        plt.close(fig)
        print(f"FRANKA_MESH_PREVIEW_DONE {POSTER} bytes={POSTER.stat().st_size}")
        return
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(fig, draw, frames=n_frames, interval=1000 / fps, blit=False)
    movie.save(VIDEO, writer=animation.FFMpegWriter(fps=fps, bitrate=5000, metadata={"title": "Franka v3c teacher rollout"}))
    plt.close(fig)
    print(f"FRANKA_MESH_VIDEO_DONE {VIDEO} frames={n_frames} fps={fps} bytes={VIDEO.stat().st_size}")
    print(f"FRANKA_MESH_PREVIEW_DONE {POSTER} bytes={POSTER.stat().st_size}")


if __name__ == "__main__":
    main()
