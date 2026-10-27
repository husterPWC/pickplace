"""Simulator-free geometry helpers for fixed-line conveyor interception."""

from __future__ import annotations

import torch


def fixed_hover_delta(
    tcp_position: torch.Tensor,
    grasp_position_delta: torch.Tensor,
    intercept_x: float,
    hover_height: float,
) -> torch.Tensor:
    """Vector from TCP to the fixed interception hover point.

    X is fixed in the cell, Y and Z come from current object/grasp geometry.
    No velocity, contact, force, or tactile quantity is used.
    """
    if tcp_position.shape[-1] != 3 or grasp_position_delta.shape != tcp_position.shape:
        raise ValueError("tcp_position and grasp_position_delta must have matching (..., 3) shapes")
    delta = grasp_position_delta.clone()
    delta[..., 0] = float(intercept_x) - tcp_position[..., 0]
    delta[..., 2] += float(hover_height)
    return delta


def progressive_intercept_hover_delta(
    tcp_position: torch.Tensor,
    grasp_position_delta: torch.Tensor,
    intercept_x: float,
    intercept_lead_distance: float,
    intercept_align_lead_distance: float,
    hover_height: float,
    pre_descent_height: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move down toward a fixed interception line before final alignment.

    The schedule is geometric rather than velocity-driven.  While the object
    travels from ``intercept_x - intercept_lead_distance`` to
    ``intercept_x - intercept_align_lead_distance``, the target clearance
    decreases linearly from ``hover_height`` to ``pre_descent_height``.  X
    remains fixed at the interception line, so the arm meets the object there
    instead of reactively chasing it from behind.
    """
    if not 0.0 < float(intercept_align_lead_distance) < float(intercept_lead_distance):
        raise ValueError("align lead distance must be positive and smaller than intercept lead distance")
    if not 0.0 < float(pre_descent_height) < float(hover_height):
        raise ValueError("pre-descent height must be positive and below hover height")
    target_x = tcp_position[..., 0] + grasp_position_delta[..., 0]
    descent_start_x = float(intercept_x) - float(intercept_lead_distance)
    descent_end_x = float(intercept_x) - float(intercept_align_lead_distance)
    progress = ((target_x - descent_start_x) / (descent_end_x - descent_start_x)).clamp(0.0, 1.0)
    clearance = float(hover_height) + progress * (
        float(pre_descent_height) - float(hover_height)
    )
    delta = grasp_position_delta.clone()
    delta[..., 0] = float(intercept_x) - tcp_position[..., 0]
    delta[..., 2] += clearance
    return delta, progress


def interception_objective(
    tcp_position: torch.Tensor,
    grasp_pose_delta: torch.Tensor,
    intercept_x: float,
    intercept_lead_distance: float,
    hover_height: float,
    intercept_align_lead_distance: float = 0.04,
    pre_descent_height: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return active distance, near-grasp stage mask, and hover delta.

    Before the object reaches the lead line the distance is purely positional
    distance to the fixed hover point.  Afterwards it is translation distance
    to the grasp pose; orientation is deliberately not mixed into either
    distance and can be shaped separately only in the near stage.
    """
    if grasp_pose_delta.shape[-1] != 6:
        raise ValueError("grasp_pose_delta must have shape (..., 6)")
    target_x = tcp_position[..., 0] + grasp_pose_delta[..., 0]
    if pre_descent_height is None:
        hover = fixed_hover_delta(
            tcp_position, grasp_pose_delta[..., :3], intercept_x, hover_height
        )
        near = target_x >= float(intercept_x) - float(intercept_lead_distance)
    else:
        hover, _ = progressive_intercept_hover_delta(
            tcp_position,
            grasp_pose_delta[..., :3],
            intercept_x,
            intercept_lead_distance,
            intercept_align_lead_distance,
            hover_height,
            pre_descent_height,
        )
        near = target_x >= float(intercept_x) - float(intercept_align_lead_distance)
    hover_distance = torch.linalg.vector_norm(hover, dim=-1)
    grasp_distance = torch.linalg.vector_norm(grasp_pose_delta[..., :3], dim=-1)
    return torch.where(near, grasp_distance, hover_distance), near, hover


def signed_distance_progress(
    previous_distance: torch.Tensor,
    current_distance: torch.Tensor,
    comparable: torch.Tensor | None = None,
) -> torch.Tensor:
    """Signed one-step progress: positive toward, negative away."""
    if previous_distance.shape != current_distance.shape:
        raise ValueError("previous_distance and current_distance shapes must match")
    progress = previous_distance - current_distance
    if comparable is not None:
        progress = torch.where(comparable, progress, torch.zeros_like(progress))
    return progress


def multiscale_position_quality(
    distance: torch.Tensor,
    near_scale: float = 0.05,
    far_scale: float = 0.35,
    near_weight: float = 0.3,
) -> torch.Tensor:
    """Position-only counterpart of the task's multiscale pose quality."""
    if min(float(near_scale), float(far_scale)) <= 0.0:
        raise ValueError("position quality scales must be positive")
    if not 0.0 <= float(near_weight) <= 1.0:
        raise ValueError("near_weight must be in [0, 1]")
    near = 1.0 - torch.tanh(distance / float(near_scale))
    far = 1.0 - torch.tanh(distance / float(far_scale))
    return float(near_weight) * near + (1.0 - float(near_weight)) * far


def progress_augmented_step_reward(
    quality: torch.Tensor,
    previous_distance: torch.Tensor,
    current_distance: torch.Tensor,
    step_dt: float,
    progress_gain: float,
    comparable: torch.Tensor | None = None,
) -> torch.Tensor:
    """Raw dense reward whose dt-scaled progress is exactly k*(d0-d1)."""
    if step_dt <= 0.0 or progress_gain < 0.0:
        raise ValueError("step_dt must be positive and progress_gain nonnegative")
    return quality + float(progress_gain) * signed_distance_progress(
        previous_distance, current_distance, comparable
    ) / float(step_dt)


def outside_workspace(
    position: torch.Tensor,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> torch.Tensor:
    """Per-row mask for an axis-aligned Cartesian safety workspace."""
    if position.shape[-1] != 3:
        raise ValueError("position must have shape (..., 3)")
    lo = torch.as_tensor(lower, dtype=position.dtype, device=position.device)
    hi = torch.as_tensor(upper, dtype=position.dtype, device=position.device)
    if bool((lo >= hi).any()):
        raise ValueError("workspace lower bounds must be below upper bounds")
    return ((position < lo) | (position > hi)).any(dim=-1)


def update_retreat_count(
    count: torch.Tensor,
    previous_distance: torch.Tensor,
    current_distance: torch.Tensor,
    comparable: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Increment only for consecutive comparable steps moving away."""
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    retreating = comparable & (current_distance > previous_distance + float(epsilon))
    return torch.where(retreating, count + 1, torch.zeros_like(count))


def moving_intercept_retreat_comparable(
    initialized: torch.Tensor,
    previous_near: torch.Tensor,
    current_near: torch.Tensor,
    target_x: torch.Tensor,
    descent_start_x: float | None,
) -> torch.Tensor:
    """Return rows where distance growth means actual unsafe retreat.

    With a fixed target, consecutive distance growth is a useful escape
    detector.  Once a progressive interception target starts moving, however,
    that same signal mixes target motion with robot retreat.  The conveyor
    deadline and Cartesian workspace remain active during interception, so the
    ambiguous retreat counter is suspended only after planned pre-descent has
    begun.
    """
    comparable = initialized & current_near.eq(previous_near)
    if descent_start_x is not None:
        comparable &= target_x < float(descent_start_x)
    return comparable
