#!/usr/bin/env python3
"""Validate the frozen G1 checkpoint and print a simulator-free G2 plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pickplace.g2_collection import (
    JAKA_VISUAL_TASKS,
    estimated_shard_bytes,
    profile,
    validate_g1_teacher_payload,
    validate_jaka_visual_teacher_payload,
)


DEFAULT_CHECKPOINT = Path(
    "artifacts/moving_sphere_static_bowl/teachers/"
    "moving_sphere_g1_place_008_1024_no_rtx/checkpoints/ppo_teacher_262144.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--env-checkpoint", type=Path, default=None)
    parser.add_argument("--jaka-visual-task", choices=sorted(JAKA_VISUAL_TASKS), default=None)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--image-size", type=int, default=84)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    env_checkpoint = args.env_checkpoint or args.checkpoint
    env_payload = torch.load(env_checkpoint, map_location="cpu", weights_only=False)
    contract = (
        validate_jaka_visual_teacher_payload(payload, env_payload, args.jaka_visual_task)
        if args.jaka_visual_task
        else validate_g1_teacher_payload(payload)
    )
    selected = profile(args.profile)
    size = estimated_shard_bytes(selected["frames"], args.image_size)
    report = {
        "stage": "G2",
        "checkpoint": str(args.checkpoint.resolve()),
        "env_checkpoint": str(env_checkpoint.resolve()),
        "jaka_visual_task": args.jaka_visual_task,
        "profile": args.profile,
        **selected,
        "image_size": args.image_size,
        "include_depth": True,
        "store_teacher_inputs": True,
        "estimated_size_bytes": size,
        "estimated_size_gb": round(size / 1024**3, 2),
        "contract": contract,
    }
    print("G2_COLLECTION_PLAN " + json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
