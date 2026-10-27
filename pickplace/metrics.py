"""Simulator-free training metrics. Imports only torch."""

from __future__ import annotations

from collections.abc import Mapping

import torch

OUTCOME_TERMS: tuple[str, ...] = (
    "success",
    "bowl_exited_zone",
    "bowl_off_belt",
    "bowl_tipped",
    "food_off_table",
    "time_out",
)

MOVING_CUBE_OUTCOME_TERMS: tuple[str, ...] = (
    "success",
    "cube_exited_zone",
    "cube_off_belt",
    "time_out",
)

MOVING_SPHERE_RESIDUAL_OUTCOME_TERMS: tuple[str, ...] = (
    "success",
    "cube_exited_zone",
    "cube_off_belt",
    "tcp_escape",
    "time_out",
)

TASK_OUTCOME_TERMS: dict[str, tuple[str, ...]] = {
    "FoodRobot-Cell-v0": OUTCOME_TERMS,
    "FoodRobot-StaticSphereMovingBowl-v0": OUTCOME_TERMS,
    "FoodRobot-MovingCubeStaticBowl-v0": MOVING_CUBE_OUTCOME_TERMS,
    "FoodRobot-MovingCubeLift-v0": MOVING_CUBE_OUTCOME_TERMS,
    # Outcome field names remain stable for dataset/checkpoint compatibility;
    # in sphere tasks ``cube_*`` means the generic moving food rigid body.
    "FoodRobot-MovingSphereStaticBowl-v0": MOVING_CUBE_OUTCOME_TERMS,
    "FoodRobot-MovingSphereLift-v0": MOVING_CUBE_OUTCOME_TERMS,
    "FoodRobot-MovingSphereGapLift-v0": MOVING_CUBE_OUTCOME_TERMS,
    "FoodRobot-MovingSphereCanonicalGapLift-v0": MOVING_CUBE_OUTCOME_TERMS,
    "FoodRobot-MovingSphereA0Lift-v0": MOVING_CUBE_OUTCOME_TERMS,
    "FoodRobot-MovingSphereResidualGapLift-v0": MOVING_SPHERE_RESIDUAL_OUTCOME_TERMS,
}


def outcome_terms(task: str) -> tuple[str, ...]:
    """Outcome vocabulary for a registered task, without changing the legacy default."""
    try:
        return TASK_OUTCOME_TERMS[task]
    except KeyError as error:
        raise KeyError(f"Unknown task {task!r}. Available: {sorted(TASK_OUTCOME_TERMS)}") from error


def outcome_rates(done: torch.Tensor, outcomes: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Fraction of finished episodes that ended with each outcome.

    Args:
        done: boolean tensor marking the last step of each finished episode, any shape.
        outcomes: outcome name -> boolean tensor of the same shape as ``done``.

    Returns:
        name -> (outcome & done).sum() / done.sum(); an empty dict if no episode finished.
    """
    finished = done.reshape(-1).bool()
    n = int(finished.sum())
    if n == 0:
        return {}
    return {name: float((flags.reshape(-1).bool() & finished).sum()) / n for name, flags in outcomes.items()}
