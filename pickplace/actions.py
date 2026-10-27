"""Project-specific Isaac Lab action terms."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass


class CoupledGripperPositionAction(ActionTerm):
    """One continuous action controlling both symmetric finger joints.

    Input ``-1`` is fully closed and ``+1`` fully open.  Using a single
    coordinate makes mechanically impossible asymmetric finger commands
    unavailable to the policy.
    """

    cfg: "CoupledGripperPositionActionCfg"

    def __init__(self, cfg: "CoupledGripperPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        # Isaac Lab 3 can expose the articulation through the isaaclab_physx
        # implementation class rather than the public re-export, so use its
        # required interface instead of a brittle cross-package isinstance.
        if not hasattr(self._asset, "find_joints") or not hasattr(self._asset, "set_joint_position_target"):
            raise TypeError("CoupledGripperPositionAction requires an articulation-like asset")
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        if len(self._joint_ids) < 1:
            raise ValueError(f"No gripper joints matched {cfg.joint_names}")
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions.clamp(-1.0, 1.0))
        midpoint = 0.5 * (float(self.cfg.open_pos) + float(self.cfg.closed_pos))
        half_range = 0.5 * (float(self.cfg.open_pos) - float(self.cfg.closed_pos))
        target = midpoint + half_range * self._raw_actions
        self._processed_actions.copy_(target.expand(-1, len(self._joint_ids)))

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 1.0
        self._processed_actions[env_ids] = float(self.cfg.open_pos)


@configclass
class CoupledGripperPositionActionCfg(ActionTermCfg):
    class_type: type = CoupledGripperPositionAction
    asset_name: str = "robot"
    joint_names: list[str] = []
    open_pos: float = 0.04
    closed_pos: float = 0.0


class PersistentDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Relative differential IK whose zero action holds the last commanded pose.

    Upstream relative IK re-latches the measured pose on every policy step.
    That is harmless for very stiff stock assets, but lets a heavy industrial
    arm's small servo tracking error accumulate as task-space sag.  This term
    integrates deltas from the previous command pose and therefore models the
    pose-hold behavior of a real robot controller.
    """

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._goal_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        ee_pos, ee_quat = self._compute_frame_pose()
        fresh = ~self._goal_initialized
        self._goal_pos[fresh] = ee_pos[fresh]
        self._goal_quat[fresh] = ee_quat[fresh]
        self._goal_initialized[fresh] = True
        # Non-zero policy deltas retain upstream velocity-like semantics
        # (measured pose + one delta).  Only a true zero command holds the
        # previous target.  Blindly accumulating a servo's repeated error
        # command would wind up the target while the physical arm catches up.
        moving = torch.linalg.vector_norm(self._processed_actions, dim=-1) > 1.0e-8
        base_pos = torch.where(moving[:, None], ee_pos, self._goal_pos)
        base_quat = torch.where(moving[:, None], ee_quat, self._goal_quat)
        self._goal_pos, self._goal_quat = math_utils.apply_delta_pose(
            base_pos, base_quat, self._processed_actions
        )
        # The controller is relative, so a zero delta based at the integrated
        # goal copies that exact pose into its desired-pose buffers.
        zero = torch.zeros_like(self._processed_actions)
        self._ik_controller.set_command(zero, self._goal_pos, self._goal_quat)

    def _compute_frame_jacobian(self) -> torch.Tensor:
        """Return the TCP Jacobian with the offset expressed in base axes.

        The bundled action term applies ``skew(offset)`` before rotating the
        body-local offset and also rotates angular velocity by the tool
        rotation.  That is only benign for short axis-aligned tools.  AG95's
        167.56 mm rotated TCP requires the rigid-body Jacobian relation in a
        single (robot-base) coordinate frame.
        """
        jacobian = self.jacobian_b.clone()
        if self.cfg.body_offset is None:
            return jacobian
        body_quat_w = self._asset.data.body_quat_w.torch[:, self._body_idx]
        root_quat_w = self._asset.data.root_quat_w.torch
        body_quat_b = math_utils.quat_mul(math_utils.quat_inv(root_quat_w), body_quat_w)
        offset_b = torch.bmm(
            math_utils.matrix_from_quat(body_quat_b), self._offset_pos.unsqueeze(-1)
        ).squeeze(-1)
        jacobian[:, 0:3, :] += torch.bmm(
            -math_utils.skew_symmetric_matrix(offset_b), jacobian[:, 3:, :]
        )
        return jacobian

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._goal_initialized.zero_()
        else:
            self._goal_initialized[env_ids] = False


@configclass
class PersistentDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = PersistentDifferentialInverseKinematicsAction
