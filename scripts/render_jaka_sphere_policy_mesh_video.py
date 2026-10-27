#!/usr/bin/env python3
"""CPU mesh rendering for a recorded JAKA moving-sphere policy rollout."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

from render_zu5_ag95_mesh_video import load_visuals, shaded


def _options(argv: list[str]) -> dict[str, str]:
    result = {}
    for item in argv:
        if "=" not in item:
            raise SystemExit("usage: render_jaka_sphere_policy_mesh_video.py trace=... [out=...]")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def box_triangles(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    half = size / 2.0
    vertices = np.asarray(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float
    ) * half + center
    faces = np.asarray(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ],
        dtype=np.int64,
    )
    return vertices[faces]


opts = _options(sys.argv[1:])
if "trace" not in opts:
    raise SystemExit("trace=<trajectory.npz> is required")
trace_path = Path(opts["trace"]).resolve()
video_path = Path(opts.get("out", str(trace_path.with_suffix(".mp4")))).resolve()
preview_path = video_path.with_name(f"{video_path.stem}_first_frame.png")
data = np.load(trace_path)
body_index = {str(name): index for index, name in enumerate(data["body_names"])}
visuals = [visual for visual in load_visuals(max_faces=480) if visual.link in body_index]

fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
fig.patch.set_facecolor("#10151b")
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
ax.set_facecolor("#10151b")
collections: list[tuple[Poly3DCollection, object]] = []
for visual in visuals:
    collection = Poly3DCollection([], linewidths=0.015, edgecolors=(0, 0, 0, 0.06))
    collection.set_zorder(4)
    ax.add_collection3d(collection)
    collections.append((collection, visual))

belt_collection = Poly3DCollection([], linewidths=0.1)
pallet_collection = Poly3DCollection([], linewidths=0.1)
sphere_collection = Poly3DCollection([], linewidths=0.0, edgecolors="none")
bowl_collection = Poly3DCollection([], linewidths=0.1)
for collection in (belt_collection, pallet_collection, sphere_collection, bowl_collection):
    ax.add_collection3d(collection)
belt_collection.set_zorder(1)
pallet_collection.set_zorder(2)
bowl_collection.set_zorder(3)
sphere_collection.set_zorder(9)

# Use a triangulated sphere so the output remains a true solid scene rather
# than the earlier skeleton-style diagnostic.
phi = np.linspace(0.0, math.pi, 13)
theta = np.linspace(0.0, 2.0 * math.pi, 25)
sphere_vertices = np.asarray(
    [[0.02 * math.sin(p) * math.cos(t), 0.02 * math.sin(p) * math.sin(t), 0.02 * math.cos(p)]
     for p in phi for t in theta]
)
sphere_faces = []
cols = len(theta)
for i in range(len(phi) - 1):
    for j in range(len(theta) - 1):
        a, b = i * cols + j, (i + 1) * cols + j
        sphere_faces.extend(((a, b, b + 1), (a, b + 1, a + 1)))
sphere_faces = np.asarray(sphere_faces, dtype=np.int64)

title = fig.text(0.04, 0.94, "JAKA ZU5 + AG95 | deterministic J0-S1 rollout", color="white", fontsize=17, weight="bold")
status = fig.text(0.04, 0.89, "", color="#e8edf2", fontsize=12)
note = fig.text(0.04, 0.85, "Recorded Isaac physics; CPU mesh replay; policy camera input disabled", color="#aeb9c4", fontsize=10)

ax.set_xlim(-0.02, 0.78)
ax.set_ylim(-0.18, 0.46)
ax.set_zlim(0.0, 0.70)
ax.set_box_aspect((0.80, 0.64, 0.70))
# Look from the conveyor side.  The opposite azimuth puts the ZU5 shoulder in
# front of both AG95 and the sphere and makes a solid-mesh video look skeletal.
ax.view_init(elev=float(opts.get("elev", 24)), azim=float(opts.get("azim", 90)))
ax.set_proj_type("persp", focal_length=0.9)
ax.set_xlabel("X / m", color="white")
ax.set_ylabel("Y / m", color="white")
ax.set_zlabel("Z / m", color="white")
ax.tick_params(colors="#c6cbd1")
ax.grid(True, color="#35404c", alpha=0.40)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((0.16, 0.18, 0.21, 1.0))

belt_center = np.asarray([float(data["belt_center"][0]), float(data["belt_center"][1]), 0.016])
belt_triangles = box_triangles(
    belt_center,
    np.asarray([float(data["belt_visual_length"]), 2.0 * float(data["belt_half_width"]), 0.008]),
)
belt_collection.set_verts(belt_triangles)
belt_collection.set_facecolors(shaded(np.asarray(to_rgb("#252b31")), belt_triangles))

# Static destination bowl: an annular wall plus a thin base.  It is drawn from
# the recorded rigid-body pose so its placement matches the Isaac episode.
bowl_center = data["bowl_pos"][0]
segments = 32
inner_radius, outer_radius = 0.07, 0.076
z0, z1 = float(bowl_center[2]), float(bowl_center[2] + 0.056)
bowl_vertices = []
for i in range(segments):
    angle = 2.0 * math.pi * i / segments
    for radius in (inner_radius, outer_radius):
        bowl_vertices.extend(
            ([bowl_center[0] + radius * math.cos(angle), bowl_center[1] + radius * math.sin(angle), z0],
             [bowl_center[0] + radius * math.cos(angle), bowl_center[1] + radius * math.sin(angle), z1])
        )
bowl_vertices = np.asarray(bowl_vertices)
bowl_faces = []
for i in range(segments):
    j = (i + 1) % segments
    ii, jj = 4 * i, 4 * j
    # inner side, outer side, and top annulus
    bowl_faces.extend(
        ((ii, jj, jj + 1), (ii, jj + 1, ii + 1),
         (ii + 2, ii + 3, jj + 3), (ii + 2, jj + 3, jj + 2),
         (ii + 1, jj + 1, jj + 3), (ii + 1, jj + 3, ii + 3))
    )
bowl_triangles = bowl_vertices[np.asarray(bowl_faces, dtype=np.int64)]
bowl_base = box_triangles(
    np.asarray([bowl_center[0], bowl_center[1], z0 + 0.003]),
    np.asarray([2.0 * outer_radius, 2.0 * outer_radius, 0.006]),
)
bowl_triangles = np.concatenate((bowl_triangles, bowl_base), axis=0)
bowl_collection.set_verts(bowl_triangles)
bowl_collection.set_facecolors(shaded(np.asarray(to_rgb("#d9d7c7")), bowl_triangles))


def draw(frame: int):
    positions = data["body_pos"][frame]
    quaternions = data["body_quat"][frame]
    artists = []
    for collection, visual in collections:
        index = body_index[visual.link]
        triangles = Rotation.from_quat(quaternions[index]).apply(
            visual.triangles.reshape(-1, 3)
        ).reshape(-1, 3, 3)
        triangles += positions[index]
        collection.set_verts(triangles)
        collection.set_facecolors(shaded(visual.color, triangles))
        artists.append(collection)

    # The moving pallet plate is the last rigid body of the two-body prismatic articulation.
    pallet_center = data["pallet_pos"][frame, -1].copy()
    pallet_center[2] += 0.005
    pallet_triangles = box_triangles(pallet_center, np.asarray([0.22, 0.26, 0.01]))
    pallet_collection.set_verts(pallet_triangles)
    pallet_collection.set_facecolors(shaded(np.asarray(to_rgb("#58616a")), pallet_triangles))

    sphere_triangles = sphere_vertices[sphere_faces] + data["sphere_pos"][frame]
    sphere_collection.set_verts(sphere_triangles)
    sphere_collection.set_facecolors(shaded(np.asarray(to_rgb("#43d96b")), sphere_triangles))
    phase = str(data["phase"][frame])
    error_deg = math.degrees(float(data["pose_rotation_error"][frame]))
    status.set_text(
        f"t={float(data['time'][frame]):5.2f} s    phase={phase:<6}    "
        f"top-down target rotation error={error_deg:.4f} deg"
    )
    return artists + [belt_collection, pallet_collection, sphere_collection, bowl_collection, title, status, note]


draw(0)
video_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(preview_path, dpi=100, facecolor=fig.get_facecolor())
if opts.get("preview_only", "false").lower() in ("1", "true", "yes"):
    plt.close(fig)
    print(f"JAKA_POLICY_FIRST_FRAME {preview_path} ({preview_path.stat().st_size} bytes)")
    raise SystemExit(0)
fps = max(1, round(1.0 / float(data["dt"])))
movie = animation.FuncAnimation(fig, draw, frames=len(data["time"]), interval=1000 / fps, blit=False)
movie.save(video_path, writer=animation.FFMpegWriter(fps=fps, bitrate=5000, metadata={"title": "JAKA J0-S1 top-down validation"}))
plt.close(fig)
print(f"JAKA_POLICY_VIDEO {video_path} ({video_path.stat().st_size} bytes)")
print(f"JAKA_POLICY_FIRST_FRAME {preview_path} ({preview_path.stat().st_size} bytes)")
