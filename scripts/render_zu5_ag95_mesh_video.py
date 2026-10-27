#!/usr/bin/env python3
"""Render the recorded ZU5+AG95 PhysX trajectory with the actual STL meshes.

This renderer is intentionally independent of Isaac RTX.  It reads link poses
from ``trajectory.npz``, applies them to the URDF visual meshes, and writes an
H.264 video that works with the workstation's current display driver.
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "zu5_ag95_validation"
TRAJECTORY = OUT / "trajectory.npz"
URDF = ROOT / "assets" / "models" / "robots" / "Zu5Ag95" / "zu5_ag95.urdf"
CLOSEUP = "--closeup" in sys.argv[1:]
VIDEO = OUT / ("zu5_ag95_gripper_closeup.mp4" if CLOSEUP else "zu5_ag95_mesh_validation.mp4")
POSTER = OUT / ("gripper_closeup_preview.png" if CLOSEUP else "mesh_preview.png")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402


@dataclass
class VisualMesh:
    link: str
    triangles: np.ndarray
    color: np.ndarray


def vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    return np.asarray([float(x) for x in text.split()] if text else default, dtype=np.float64)


def load_visuals(max_faces: int = 550) -> list[VisualMesh]:
    root = ET.parse(URDF).getroot()
    materials: dict[str, np.ndarray] = {}
    for material in root.findall("material"):
        rgba = material.find("color")
        if material.get("name") and rgba is not None:
            materials[material.get("name")] = vector(rgba.get("rgba"), (0.7, 0.7, 0.7, 1.0))[:3]

    visuals = []
    for link in root.findall("link"):
        link_name = link.get("name")
        for visual in link.findall("visual"):
            mesh_node = visual.find("geometry/mesh")
            if mesh_node is None:
                continue
            path = (URDF.parent / mesh_node.get("filename")).resolve()
            mesh = trimesh.load_mesh(path, process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            scale = vector(mesh_node.get("scale"), (1.0, 1.0, 1.0))
            vertices *= scale
            origin = visual.find("origin")
            xyz = vector(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            rpy = vector(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            vertices = Rotation.from_euler("xyz", rpy).apply(vertices) + xyz
            faces = np.asarray(mesh.faces, dtype=np.int64)
            if len(faces) > max_faces:
                # A sparse subset of faces looks like a wire cloud.  Build a
                # closed, low-polygon outer shell from surface samples instead;
                # it keeps each link solid and recognizable without RTX.
                local_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                samples, _ = trimesh.sample.sample_surface(local_mesh, 420)
                shell = trimesh.Trimesh(vertices=samples, process=False).convex_hull
                triangles = np.asarray(shell.vertices)[np.asarray(shell.faces)]
            else:
                triangles = vertices[faces]
            material = visual.find("material")
            color = materials.get(material.get("name") if material is not None else "", np.array([0.65] * 3))
            visuals.append(VisualMesh(link_name, triangles, np.asarray(color)))
    return visuals


def cube_triangles(center: np.ndarray) -> np.ndarray:
    half = np.asarray([0.015, 0.015, 0.020])
    v = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float) * half + center
    faces = np.asarray(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ],
        dtype=np.int64,
    )
    return v[faces]


def shaded(base: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.maximum(norm, 1e-9)
    light = np.asarray([-0.4, -0.5, 0.75])
    light /= np.linalg.norm(light)
    intensity = 0.48 + 0.52 * np.abs(normal @ light)
    return np.clip(base[None, :] * intensity[:, None] + 0.08, 0.0, 1.0)


def main() -> None:
    data = np.load(TRAJECTORY)
    body_names = list(data["body_names"])
    body_index = {name: i for i, name in enumerate(body_names)}
    joint_index = list(data["joint_names"]).index("left_outer_knuckle_joint")
    visuals = [visual for visual in load_visuals(max_faces=900 if CLOSEUP else 550) if visual.link in body_index]
    if CLOSEUP:
        visuals = [visual for visual in visuals if visual.link == "Link_6" or visual.link.startswith(("ag95_", "left_", "right_"))]

    # Half the stored frames: a compact 7.8 s playback still includes every
    # test phase while keeping CPU mesh rendering practical.
    frame_ids = np.arange(0, len(data["time"]), 2)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#10151b")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#10151b")

    collections: list[tuple[Poly3DCollection, VisualMesh]] = []
    for visual in visuals:
        collection = Poly3DCollection([], linewidths=0.02, edgecolors=(0, 0, 0, 0.08))
        ax.add_collection3d(collection)
        collections.append((collection, visual))
    cube_collection = Poly3DCollection([], linewidths=0.6, edgecolors="#b8ffc8")
    ax.add_collection3d(cube_collection)

    phase_labels = {
        "open": "OPEN",
        "free_close": "FREE CLOSE",
        "reopen": "REOPEN",
        "grasp_close": "CONTACT GRASP",
        "hold": "HOLD",
        "loaded_hold": "1 N DOWNWARD LOAD",
        "release": "RELEASE",
    }
    title = fig.text(
        0.05,
        0.93,
        "AG95 close-up | measured PhysX mesh replay" if CLOSEUP else "ZU5 + AG95 | measured PhysX mesh replay",
        color="white",
        fontsize=17,
        weight="bold",
    )
    status = fig.text(0.05, 0.875, "", color="#e8edf2", fontsize=12)
    note = fig.text(
        0.05,
        0.83,
        "Actual STL geometry transformed by the recorded link poses (no RTX required)",
        color="#aeb9c4",
        fontsize=10,
    )

    def setup_axes() -> None:
        if CLOSEUP:
            ax.set_xlim(0.68, 0.91)
            ax.set_ylim(-0.26, 0.06)
            ax.set_zlim(0.40, 0.61)
            ax.set_box_aspect((0.23, 0.32, 0.21))
            # Look from the fingertip side so the left/right fingers do not
            # overlap the gripper body in projection.
            ax.view_init(elev=18, azim=-90)
        else:
            ax.set_xlim(-0.06, 0.93)
            ax.set_ylim(-0.33, 0.14)
            ax.set_zlim(0.39, 0.78)
            ax.set_box_aspect((0.99, 0.47, 0.39))
            ax.view_init(elev=23, azim=119)
        ax.set_proj_type("persp", focal_length=0.9)
        ax.set_xlabel("X / m", color="white")
        ax.set_ylabel("Y / m", color="white")
        ax.set_zlabel("Z / m", color="white")
        ax.tick_params(colors="#c6cbd1")
        ax.grid(True, color="#35404c", alpha=0.45)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.16, 0.18, 0.21, 1.0))

    setup_axes()

    def draw(sequence_index: int):
        frame = int(frame_ids[sequence_index])
        positions = data["body_pos"][frame]
        quaternions = data["body_quat"][frame]  # Isaac Lab 3: xyzw
        artists = []
        for collection, visual in collections:
            idx = body_index[visual.link]
            tri = Rotation.from_quat(quaternions[idx]).apply(visual.triangles.reshape(-1, 3)).reshape(-1, 3, 3)
            tri += positions[idx]
            collection.set_verts(tri)
            collection.set_facecolors(shaded(visual.color, tri))
            artists.append(collection)

        cube_pos = data["cube_pos"][frame]
        if cube_pos[2] > 0.0:
            cube_tri = cube_triangles(cube_pos)
            cube_collection.set_verts(cube_tri)
            cube_collection.set_facecolors(shaded(np.asarray(to_rgb("#35d967")), cube_tri))
        else:
            cube_collection.set_verts([])
        artists.append(cube_collection)

        left, right = data["pad_left"][frame], data["pad_right"][frame]
        gap = np.linalg.norm(left - right) * 1000.0
        phase = str(data["phase"][frame])
        q = data["joint_pos"][frame, joint_index]
        force = data["contact_force"][frame]
        status.set_text(
            f"{phase_labels.get(phase, phase)}    master={q:.3f} rad    pad gap={gap:.1f} mm    "
            f"net contact={force:.2f} N"
        )
        return artists + [title, status, note]

    # Save a representative grasp/hold image in addition to the MP4.
    hold_frames = np.where(data["phase"] == "hold")[0]
    poster_frame = int(hold_frames[len(hold_frames) // 2])
    poster_seq = int(np.argmin(np.abs(frame_ids - poster_frame)))
    draw(poster_seq)
    fig.savefig(POSTER, dpi=100, facecolor=fig.get_facecolor())

    movie = animation.FuncAnimation(fig, draw, frames=len(frame_ids), interval=1000 / 15, blit=False)
    writer = animation.FFMpegWriter(fps=15, bitrate=4500, metadata={"title": "ZU5 AG95 PhysX mesh validation"})
    movie.save(VIDEO, writer=writer)
    plt.close(fig)
    print(f"MESH_VIDEO_DONE {VIDEO} ({VIDEO.stat().st_size} bytes)")
    print(f"MESH_PREVIEW_DONE {POSTER} ({POSTER.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
