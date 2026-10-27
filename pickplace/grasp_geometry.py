"""Geometry used by the speed-free sphere grasp task.

The convention is deliberately explicit: each candidate frame has local
``+z`` pointing along the gripper approach direction (downwards), local
``+y`` along finger closing, and local ``+x`` along the finger length.  Poses
are represented by a position and a 3x3 rotation whose columns are those
three axes in the cell frame.
"""

from __future__ import annotations

import math

import torch


def quat_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert normalized-or-not xyzw quaternions to rotation matrices."""
    q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        (
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ),
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)


def rotation_matrix_to_rotvec(matrix: torch.Tensor) -> torch.Tensor:
    """Stable logarithm map SO(3) -> axis-angle rotation vector."""
    skew = torch.stack(
        (matrix[..., 2, 1] - matrix[..., 1, 2],
         matrix[..., 0, 2] - matrix[..., 2, 0],
         matrix[..., 1, 0] - matrix[..., 0, 1]),
        dim=-1,
    )
    cos_angle = ((matrix.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)
    sin_angle = torch.sin(angle)
    scale = torch.where(
        angle < 1.0e-4,
        0.5 + angle.square() / 12.0,
        angle / (2.0 * sin_angle.clamp_min(1.0e-6)),
    )
    result = skew * scale.unsqueeze(-1)
    # The skew part vanishes at exactly pi. Recover an axis from the diagonal
    # instead of incorrectly reporting zero rotation for a 180-degree error.
    near_pi = cos_angle < -0.9999
    if near_pi.any():
        diagonal = matrix.diagonal(dim1=-2, dim2=-1)
        axis = torch.sqrt(((diagonal + 1.0) * 0.5).clamp_min(0.0))
        axis_y = torch.copysign(axis[..., 1], matrix[..., 0, 1] + matrix[..., 1, 0])
        axis_z = torch.copysign(axis[..., 2], matrix[..., 0, 2] + matrix[..., 2, 0])
        axis = torch.stack((axis[..., 0], axis_y, axis_z), dim=-1)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        result = torch.where(near_pi[..., None], axis * angle[..., None], result)
    return result


def sphere_grasp_frames(
    centers: torch.Tensor,
    *,
    num_candidates: int = 40,
    tcp_height_offset: float = 0.002,
    canonical_yaw: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return analytic top-down 6D sphere-grasp candidates.

    Legacy callers leave ``canonical_yaw`` unset and sample yaw.  A sphere's
    yaw is physically unobservable, however, so symmetry-aware tasks pass a
    fixed canonical yaw and repeat that same frame across all candidate slots.
    Repetition preserves the established ``40 x 20`` network contract without
    asking either simulation or vision to infer a meaningless object yaw.

    The small positive TCP offset was validated against the Franka hand
    geometry: the TCP remains near the sphere equator while the fingertips
    extend below it.  This is a *TCP target*, not the sphere centre
    masquerading as a pose.
    """
    if num_candidates < 1:
        raise ValueError("num_candidates must be >= 1")
    dtype, device = centers.dtype, centers.device
    if canonical_yaw is None:
        yaw = torch.arange(num_candidates, dtype=dtype, device=device) * (2.0 * math.pi / num_candidates)
    else:
        yaw = torch.full((num_candidates,), float(canonical_yaw), dtype=dtype, device=device)
    c, s = torch.cos(yaw), torch.sin(yaw)
    # x=finger length, y=closing direction, z=approach (down). x cross y = z.
    x_axis = torch.stack((c, s, torch.zeros_like(c)), -1)
    y_axis = torch.stack((s, -c, torch.zeros_like(c)), -1)
    z_axis = torch.zeros_like(x_axis)
    z_axis[:, 2] = -1.0
    rotations = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    positions = centers[:, None, :].expand(-1, num_candidates, -1).clone()
    positions[..., 2] += float(tcp_height_offset)
    return positions, rotations[None].expand(centers.shape[0], -1, -1, -1)


def deterministic_grasp_points(
    positions: torch.Tensor,
    rotations: torch.Tensor,
    *,
    points_per_candidate: int = 20,
    grasp_width: float = 0.044,
) -> torch.Tensor:
    """Encode every pose as deterministic anisotropic Gaussian-like points.

    An isotropic cloud would discard orientation.  The unequal standard
    deviations preserve approach and closing axes while retaining GAP-RL's
    point-set representation and fixed 40x20 tensor shape.
    """
    if points_per_candidate < 1:
        raise ValueError("points_per_candidate must be >= 1")
    idx = torch.arange(points_per_candidate, device=positions.device, dtype=positions.dtype)
    # Low-discrepancy, deterministic normal samples (Box-Muller) avoid adding
    # observation noise between otherwise identical simulator states.
    u1 = ((idx + 0.5) / points_per_candidate).clamp_min(1.0e-6)
    u2 = torch.frac(idx * 0.6180339887498949 + 0.5)
    radius = torch.sqrt(-2.0 * torch.log(u1))
    a = 2.0 * math.pi * u2
    b = 2.0 * math.pi * torch.frac(idx * 0.4142135623730950 + 0.25)
    normal = torch.stack((radius * torch.cos(a), radius * torch.sin(a), radius * torch.cos(b)), -1)
    # Finger-length, closing-width, approach-axis spreads.
    scale = positions.new_tensor((0.010, max(float(grasp_width) * 0.18, 0.004), 0.004))
    local = normal * scale
    return positions[..., None, :] + torch.einsum("...ij,pj->...pi", rotations, local)


def nearest_grasp_delta(
    tcp_position: torch.Tensor,
    tcp_quat_xyzw: torch.Tensor,
    candidate_positions: torch.Tensor,
    candidate_rotations: torch.Tensor,
    *,
    rotation_weight: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest 6D candidate and its translation/rotation error vectors."""
    translation, rotation = grasp_pose_deltas(
        tcp_position,
        tcp_quat_xyzw,
        candidate_positions,
        candidate_rotations,
    )
    score = translation.norm(dim=-1) + float(rotation_weight) * rotation.norm(dim=-1)
    nearest = score.argmin(dim=1)
    rows = torch.arange(tcp_position.shape[0], device=tcp_position.device)
    return translation[rows, nearest], rotation[rows, nearest], nearest


def grasp_pose_deltas(
    tcp_position: torch.Tensor,
    tcp_quat_xyzw: torch.Tensor,
    candidate_positions: torch.Tensor,
    candidate_rotations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return translation and world-frame rotation errors for every candidate."""
    tcp_rotation = quat_xyzw_to_matrix(tcp_quat_xyzw)
    translation = candidate_positions - tcp_position[:, None, :]
    # Isaac Lab's relative-pose controller left-multiplies the delta rotation,
    # so the command is a world-frame error: R_target R_current^T.
    relative = torch.matmul(candidate_rotations, tcp_rotation[:, None].transpose(-1, -2))
    rotation = rotation_matrix_to_rotvec(relative)
    return translation, rotation
