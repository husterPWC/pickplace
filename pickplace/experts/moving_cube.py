"""Closed-loop state expert for the moving-cube/static-bowl task.

The expert deliberately consumes the same current state quantities exposed to
the state teacher.  It uses the cube's *current* velocity for a short intercept
lead, but has no access to a future trajectory or simulator ground-truth not in
the observation.  Its purpose is task/physics validation and curriculum
bootstrap, not final policy deployment.
"""

from __future__ import annotations

import torch


PHASE_NAMES = (
    "settle",
    "intercept_hover",
    "track_descend",
    "track_close",
    "lift",
    "bowl_hover",
    "release_pose",
    "release",
    "retreat",
    "home",
    "done",
)


def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av, aw = a[..., :3], a[..., 3:4]
    bv, bw = b[..., :3], b[..., 3:4]
    xyz = aw * bv + bw * av + torch.cross(av, bv, dim=-1)
    w = aw * bw - (av * bv).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def _quat_error_axis_angle(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """World-frame shortest axis-angle taking ``current`` to ``target`` (xyzw)."""
    inverse = torch.cat((-current[..., :3], current[..., 3:4]), dim=-1)
    error = _quat_mul_xyzw(target, inverse)
    error = torch.where(error[..., 3:4] < 0.0, -error, error)
    vector = error[..., :3]
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, error[..., 3:4].clamp(min=1e-7))
    return vector / norm.clamp(min=1e-7) * angle


def _rotate_local_z_xyzw(quat: torch.Tensor) -> torch.Tensor:
    qxyz, qw = quat[..., :3], quat[..., 3:4]
    local_z = torch.zeros_like(qxyz)
    local_z[..., 2] = 1.0
    twice_cross = 2.0 * torch.cross(qxyz, local_z, dim=-1)
    return local_z + qw * twice_cross + torch.cross(qxyz, twice_cross, dim=-1)


def _topdown_axis_angle_error(quat: torch.Tensor) -> torch.Tensor:
    approach = _rotate_local_z_xyzw(quat)
    down = torch.zeros_like(approach)
    down[..., 2] = -1.0
    axis = torch.cross(approach, down, dim=-1)
    axis_norm = torch.linalg.vector_norm(axis, dim=-1, keepdim=True)
    angle = torch.atan2(axis_norm, (approach * down).sum(dim=-1, keepdim=True))
    return axis / axis_norm.clamp(min=1e-7) * angle


class MovingCubeExpert:
    """Per-environment finite-state controller for physical task validation."""

    def __init__(self, num_envs: int, device: torch.device, dt: float, action_dim: int = 7):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.action_dim = action_dim
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.phase_time = torch.zeros(num_envs, device=device)
        self.home_quat: torch.Tensor | None = None
        self.lift_target = torch.zeros(num_envs, 3, device=device)
        self.finished = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def mark_finished(self, mask: torch.Tensor) -> None:
        self.finished |= mask
        self.phase[mask] = len(PHASE_NAMES) - 1
        self.phase_time[mask] = 0.0

    def _advance(self, mask: torch.Tensor, next_phase: int) -> None:
        mask &= ~self.finished
        self.phase[mask] = next_phase
        self.phase_time[mask] = 0.0

    @staticmethod
    def _limited_position_action(error: torch.Tensor, gain: float, limit: float) -> torch.Tensor:
        action = gain * error
        norm = torch.linalg.vector_norm(action, dim=-1, keepdim=True)
        action = action * torch.clamp(limit / norm.clamp(min=1e-7), max=1.0)
        return action.clamp(-1.0, 1.0)

    def act(self, obs: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        food = obs["privileged"]["food_pos"]
        velocity = obs["privileged"]["food_lin_vel"]
        tcp = obs["proprio"]["ee_pos"]
        quat = obs["proprio"]["ee_quat"]
        bowl = obs["belt"]["bowl_pos"]
        if self.home_quat is None:
            self.home_quat = quat.clone()

        # Transitions are evaluated from the latest closed-loop observation.
        p = self.phase
        hover_target = food + velocity * 0.15
        hover_target[:, 2] = food[:, 2] + 0.09
        hover_error = torch.linalg.vector_norm(hover_target - tcp, dim=-1)
        self._advance((p == 0) & (self.phase_time >= 0.25), 1)
        self._advance((p == 1) & ((hover_error < 0.012) | (self.phase_time >= 2.0)), 2)

        descend_target = food + velocity * 0.04
        descend_target[:, 2] = food[:, 2] + 0.002
        descend_error = torch.linalg.vector_norm(descend_target - tcp, dim=-1)
        self._advance((p == 2) & ((descend_error < 0.010) | (self.phase_time >= 1.0)), 3)

        to_lift = (p == 3) & (self.phase_time >= 0.48)
        if bool(to_lift.any()):
            self.lift_target[to_lift] = tcp[to_lift]
            self.lift_target[to_lift, 2] = 0.22
        self._advance(to_lift, 4)
        lift_error = torch.linalg.vector_norm(self.lift_target - tcp, dim=-1)
        self._advance((p == 4) & ((lift_error < 0.018) | (self.phase_time >= 1.3)), 5)

        bowl_hover = bowl.clone()
        bowl_hover[:, 2] += 0.20
        bowl_hover_error = torch.linalg.vector_norm(bowl_hover - tcp, dim=-1)
        self._advance((p == 5) & ((bowl_hover_error < 0.018) | (self.phase_time >= 2.0)), 6)

        release_pose = bowl.clone()
        release_pose[:, 2] += 0.11
        release_pose_error = torch.linalg.vector_norm(release_pose - tcp, dim=-1)
        self._advance((p == 6) & ((release_pose_error < 0.012) | (self.phase_time >= 1.0)), 7)
        self._advance((p == 7) & (self.phase_time >= 0.55), 8)
        self._advance((p == 8) & ((tcp[:, 2] >= 0.21) | (self.phase_time >= 0.8)), 9)

        home = torch.tensor((0.4633, 0.0, 0.3855), device=self.device).expand_as(tcp)
        home_error = torch.linalg.vector_norm(home - tcp, dim=-1)
        self._advance((p == 9) & ((home_error < 0.025) | (self.phase_time >= 2.0)), 10)

        # Re-read after transitions so each new phase commands immediately.
        p = self.phase
        action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        action[:, -1] = 1.0  # open
        closed = (p >= 3) & (p <= 6)
        action[closed, -1] = -1.0

        target = torch.zeros_like(tcp)
        target[p == 1] = hover_target[p == 1]
        target[p == 2] = descend_target[p == 2]
        # Continue matching the conveyor during finger closure.
        close_target = food + velocity * 0.03
        close_target[:, 2] = food[:, 2] + 0.002
        target[p == 3] = close_target[p == 3]
        target[p == 4] = self.lift_target[p == 4]
        target[p == 5] = bowl_hover[p == 5]
        target[p == 6] = release_pose[p == 6]
        target[p == 7] = release_pose[p == 7]
        retreat = bowl.clone()
        retreat[:, 2] += 0.24
        target[p == 8] = retreat[p == 8]
        target[p == 9] = home[p == 9]

        active_position = (p >= 1) & (p <= 9) & ~self.finished
        gentle = (p == 4) | (p == 5) | (p == 6)
        normal = active_position & ~gentle
        action[normal, :3] = self._limited_position_action(target[normal] - tcp[normal], 4.0, 0.32)
        action[gentle, :3] = self._limited_position_action(target[gentle] - tcp[gentle], 2.0, 0.20)

        topdown = (p >= 1) & (p <= 8) & ~self.finished
        rotation = _topdown_axis_angle_error(quat)
        action[topdown, 3:6] = (4.0 * rotation[topdown]).clamp(-1.0, 1.0)
        returning = (p == 9) & ~self.finished
        home_rotation = _quat_error_axis_angle(self.home_quat, quat)
        action[returning, 3:6] = (3.0 * home_rotation[returning]).clamp(-1.0, 1.0)

        action[self.finished] = 0.0
        action[self.finished, -1] = 1.0
        self.phase_time[~self.finished] += self.dt
        return action
