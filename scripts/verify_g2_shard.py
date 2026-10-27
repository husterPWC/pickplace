#!/usr/bin/env python3
"""Simulator-free validation and RGB-D preview for a G2 collection shard.

Run this immediately after every server smoke collection.  It checks the
student/teacher boundary, image and depth tensors, deployable state, action
semantics, provenance, and writes a contact sheet without launching Isaac.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from pickplace.artifacts import sha256_file
from pickplace.datasets import load_shard, shard_manifest
from pickplace.g2_collection import (
    CALIBRATED_SCENE_KEYS,
    DEPLOYABLE_PROPRIO_KEYS,
    DEPTH_KEYS,
    FORBIDDEN_STUDENT_SIGNALS,
    RGB_KEYS,
    TEACHER_ONLY_GROUPS,
)

EXPECTED_ACTION_SEMANTICS = {
    "teacher_residual_before_environment_g1_controller",
    "teacher_residual_before_environment_jaka_skill_controller",
}


def _has(td, key) -> bool:
    return td.get(key, None) is not None


def _depth_rgb(depth: torch.Tensor, near: float = 0.05, far: float = 1.50) -> Image.Image:
    """Make clipped metric depth visually legible; red is near, blue is far."""
    x = depth.float().squeeze(-1).clamp(near, far)
    x = ((x - near) / (far - near)).clamp(0, 1)
    rgb = torch.stack(
        (
            255 * (1 - x),
            255 * (1 - (2 * x - 1).abs()),
            255 * x,
        ),
        dim=-1,
    ).round().to(torch.uint8).cpu().numpy()
    return Image.fromarray(rgb, mode="RGB")


def _rgb(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(image.cpu().numpy(), mode="RGB")


def write_preview(sample, path: Path) -> None:
    labels = ("overview RGB", "overview depth", "wrist RGB", "wrist depth")
    rows = []
    for i in range(sample.shape[0]):
        tiles = (
            _rgb(sample["pixels", "overview_rgb"][i]),
            _depth_rgb(sample["pixels", "overview_depth"][i]),
            _rgb(sample["pixels", "wrist_rgb"][i]),
            _depth_rgb(sample["pixels", "wrist_depth"][i]),
        )
        w, h = tiles[0].size
        row = Image.new("RGB", (4 * w, h + 18), "white")
        draw = ImageDraw.Draw(row)
        for col, (label, tile) in enumerate(zip(labels, tiles, strict=True)):
            row.paste(tile, (col * w, 18))
            draw.text((col * w + 2, 2), f"{label}  row={i}", fill="black")
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def verify(path: Path, samples: int, preview: Path | None) -> dict:
    manifest = shard_manifest(path)
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    check(manifest.get("depth_cameras") == ["overview_depth", "wrist_depth"], "manifest depth cameras disagree")
    check(manifest.get("depth_unit") == "metres", "depth unit is not metres")
    check(manifest.get("depth_dtype") == "float16", "depth dtype is not float16")
    check(manifest.get("teacher_inputs_stored") is True, "teacher labels were not requested")
    check(manifest.get("action_semantics") in EXPECTED_ACTION_SEMANTICS, "wrong action semantics")
    check(manifest.get("env", {}).get("frame_stack") == 1, "G2 frame_stack must equal 1")
    check(manifest.get("env", {}).get("speed_free_observations") is True, "teacher is not speed-free")

    contract = manifest.get("g2_contract") or {}
    expected_runtime = {
        tuple(key) for key in (*RGB_KEYS, *DEPTH_KEYS, *DEPLOYABLE_PROPRIO_KEYS, *CALIBRATED_SCENE_KEYS)
    }
    actual_runtime = {tuple(key) for key in contract.get("student_runtime_keys", [])}
    check(actual_runtime == expected_runtime, "manifest student runtime keys disagree with the frozen G2 contract")
    check(tuple(contract.get("teacher_only_groups", ())) == TEACHER_ONLY_GROUPS, "teacher-only groups disagree")
    check(
        set(contract.get("forbidden_student_signals", ())) == set(FORBIDDEN_STUDENT_SIGNALS),
        "forbidden student-signal list disagrees",
    )
    runtime_leaves = {leaf for key in actual_runtime for leaf in key}
    check(not runtime_leaves.intersection(FORBIDDEN_STUDENT_SIGNALS), "forbidden signal appears in student keys")

    frames = int(manifest.get("frames", 0))
    check(frames > 0, "manifest has no frames")
    buffer = load_shard(path, batch_size=64)
    check(len(buffer) == frames, f"buffer length {len(buffer)} != manifest frames {frames}")
    if not frames:
        return {"path": str(path), "failures": failures}

    n = min(samples, frames)
    idx = torch.linspace(0, frames - 1, n).round().long().unique()
    sample = buffer[:][idx]
    image_size = int(manifest["image_size"])
    image_stats: dict[str, dict] = {}

    for key in RGB_KEYS:
        check(_has(sample, key), f"missing {'/'.join(key)}")
        if not _has(sample, key):
            continue
        value = sample[key]
        check(value.dtype == torch.uint8, f"{'/'.join(key)} dtype is {value.dtype}, expected uint8")
        check(tuple(value.shape[1:]) == (image_size, image_size, 3), f"{'/'.join(key)} has shape {tuple(value.shape)}")
        stat = {"min": int(value.min()), "max": int(value.max()), "std": round(float(value.float().std()), 4)}
        image_stats["/".join(key)] = stat
        check(stat["max"] > stat["min"] and stat["std"] > 0, f"{'/'.join(key)} is blank or constant")

    for key in DEPTH_KEYS:
        check(_has(sample, key), f"missing {'/'.join(key)}")
        if not _has(sample, key):
            continue
        value = sample[key]
        check(value.dtype == torch.float16, f"{'/'.join(key)} dtype is {value.dtype}, expected float16")
        check(tuple(value.shape[1:]) == (image_size, image_size, 1), f"{'/'.join(key)} has shape {tuple(value.shape)}")
        finite = torch.isfinite(value.float())
        stat = {
            "min_m": round(float(value.float().min()), 5),
            "max_m": round(float(value.float().max()), 5),
            "std_m": round(float(value.float().std()), 5),
            "finite_fraction": round(float(finite.float().mean()), 6),
            "nearer_than_3m_fraction": round(float((value < 2.99).float().mean()), 6),
        }
        image_stats["/".join(key)] = stat
        check(bool(finite.all()), f"{'/'.join(key)} contains NaN/inf")
        check(stat["min_m"] >= 0 and stat["max_m"] <= 3.001, f"{'/'.join(key)} lies outside [0, 3] m")
        check(stat["std_m"] > 0.0001, f"{'/'.join(key)} is constant")
        check(stat["nearer_than_3m_fraction"] > 0.01, f"{'/'.join(key)} sees no geometry below 3 m")

    for key in (*DEPLOYABLE_PROPRIO_KEYS, *CALIBRATED_SCENE_KEYS):
        check(_has(sample, key), f"missing deployable state {'/'.join(key)}")
        if _has(sample, key):
            check(bool(torch.isfinite(sample[key].float()).all()), f"non-finite deployable state {'/'.join(key)}")
    for group in TEACHER_ONLY_GROUPS:
        check(_has(sample, group), f"missing teacher-only label group {group}")
    check(_has(sample, ("teacher", "residual_action")), "missing teacher/residual_action")
    if _has(sample, ("teacher", "residual_action")) and _has(sample, "action"):
        check(torch.equal(sample["teacher", "residual_action"], sample["action"]), "teacher residual != stored action")

    checkpoint = Path(manifest.get("checkpoint", ""))
    check(checkpoint.is_file(), f"source checkpoint does not exist: {checkpoint}")
    if checkpoint.is_file():
        check(sha256_file(checkpoint) == manifest.get("checkpoint_sha256"), "source checkpoint sha256 mismatch")
    env_checkpoint = Path(manifest.get("env_checkpoint", manifest.get("checkpoint", "")))
    check(env_checkpoint.is_file(), f"environment checkpoint does not exist: {env_checkpoint}")
    if env_checkpoint.is_file():
        check(
            sha256_file(env_checkpoint) == manifest.get("env_checkpoint_sha256", manifest.get("checkpoint_sha256")),
            "environment checkpoint sha256 mismatch",
        )

    if preview is not None and all(_has(sample, key) for key in (*RGB_KEYS, *DEPTH_KEYS)):
        preview_idx = torch.linspace(0, sample.shape[0] - 1, min(6, sample.shape[0])).round().long().unique()
        write_preview(sample[preview_idx], preview)

    return {
        "path": str(path.resolve()),
        "frames": frames,
        "sampled": int(idx.numel()),
        "image_stats": image_stats,
        "preview": str(preview.resolve()) if preview is not None and preview.is_file() else None,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=Path)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    preview = None if args.no_preview else (args.preview or args.shard / "g2_rgbd_preview.png")
    report = verify(args.shard, max(1, args.samples), preview)
    print("G2_VERIFY " + json.dumps(report, ensure_ascii=False))
    if report["failures"]:
        raise SystemExit(1)
    print("G2_VERIFY_OK")


if __name__ == "__main__":
    main()
