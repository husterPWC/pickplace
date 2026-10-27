"""Termination terms. Positions in the cell frame; quaternions xyzw."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from pickplace.envs.mdp.observations import (
    dual_finger_contact_mask,
    ee_pos_cell,
    grasped_mask,
    gripper_close_fraction,
)
from pickplace.intercept import (
    interception_objective,
    moving_intercept_retreat_comparable,
    outside_workspace,
    update_retreat_count,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _pos_cell(env: ManagerBasedRLEnv, cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[cfg.name].data.root_pos_w.torch - env.scene.env_origins


def bowl_exited_zone(env: ManagerBasedRLEnv, zone_end_x: float, bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl")):
    """The bowl passed the end of the reach zone: the placement deadline is over."""
    return _pos_cell(env, bowl_cfg)[:, 0] > zone_end_x


def asset_exited_zone(
    env: ManagerBasedRLEnv,
    zone_end_x: float,
    ignore_above_z: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("food"),
) -> torch.Tensor:
    """A moving task object passed the downstream edge of the reachable zone.

    ``ignore_above_z`` turns this into a conveyor deadline rather than a
    workspace boundary: after lift-off, crossing the edge is not a missed
    interception anymore.
    """
    pos = _pos_cell(env, asset_cfg)
    exited = pos[:, 0] > zone_end_x
    if ignore_above_z is not None:
        exited &= pos[:, 2] <= ignore_above_z
    return exited


def asset_exited_zone_unless_dual_contact(
    env: ManagerBasedRLEnv,
    zone_end_x: float,
    force_threshold: float,
    manipulation_distance: float = 0.05,
    ignore_above_z: float | None = None,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_finger_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_finger_contact"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("food"),
) -> torch.Tensor:
    """Missed-intercept deadline that does not abort an in-progress grasp.

    Dual contact is a simulator-only termination label.  It is deliberately
    absent from the actor observation and does not itself count as success;
    a small TCP-centred manipulation envelope bridges transient contact
    flicker.  The strict lift termination still decides episode success.
    """
    exited = asset_exited_zone(env, zone_end_x, ignore_above_z, asset_cfg)
    held = dual_finger_contact_mask(env, left_sensor_cfg, right_sensor_cfg, force_threshold)
    within_manipulation_envelope = torch.linalg.vector_norm(
        _pos_cell(env, asset_cfg) - ee_pos_cell(env, ee_frame_cfg), dim=-1
    ) <= float(manipulation_distance)
    return exited & ~held & ~within_manipulation_envelope


class tcp_workspace_or_retreat(ManagerTermBase):
    """Fail on Cartesian safety escape or sustained motion away from target."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous_distance = torch.zeros(env.num_envs, device=env.device)
        self.previous_near = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.near_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.retreat_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.previous_distance.zero_()
            self.previous_near.zero_()
            self.near_latched.zero_()
            self.initialized.zero_()
            self.retreat_count.zero_()
        else:
            self.previous_distance[env_ids] = 0.0
            self.previous_near[env_ids] = False
            self.near_latched[env_ids] = False
            self.initialized[env_ids] = False
            self.retreat_count[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        workspace_lower: tuple[float, float, float] = (0.10, -0.20, 0.02),
        workspace_upper: tuple[float, float, float] = (0.75, 0.48, 0.75),
        retreat_epsilon: float = 0.0005,
        retreat_steps: int = 15,
        retreat_min_distance: float = 0.05,
        intercept_x: float = 0.45,
        intercept_lead_distance: float = 0.08,
        intercept_align_lead_distance: float = 0.04,
        hover_height: float = 0.08,
        pre_descent_height: float | None = None,
        num_candidates: int = 40,
        tcp_height_offset: float = 0.002,
        canonical_yaw: float | None = 0.0,
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if retreat_epsilon < 0.0 or retreat_steps < 1 or retreat_min_distance < 0.0:
            raise ValueError(
                "retreat_epsilon/retreat_min_distance must be nonnegative and retreat_steps >= 1"
            )
        # Local import avoids coupling every termination to grasp-generation
        # code at module import time.
        from pickplace.envs.mdp.observations import sphere_best_grasp_pose_delta

        tcp = ee_pos_cell(env, ee_frame_cfg)
        pose_delta = sphere_best_grasp_pose_delta(
            env,
            num_candidates=num_candidates,
            tcp_height_offset=tcp_height_offset,
            canonical_yaw=canonical_yaw,
            food_cfg=food_cfg,
            ee_frame_cfg=ee_frame_cfg,
        )
        distance, near, _ = interception_objective(
            tcp,
            pose_delta,
            intercept_x,
            intercept_lead_distance,
            hover_height,
            intercept_align_lead_distance,
            pre_descent_height,
        )
        target_x = tcp[:, 0] + pose_delta[:, 0]
        descent_start_x = (
            None
            if pre_descent_height is None
            else float(intercept_x) - float(intercept_lead_distance)
        )
        comparable = moving_intercept_retreat_comparable(
            self.initialized,
            self.previous_near,
            near,
            target_x,
            descent_start_x,
        )
        # This guard is a coarse-interception safety check, not a grasp/lift
        # failure detector.  Once the TCP is in the near manipulation basin,
        # closing and lifting intentionally change the sphere-relative pose;
        # treating that geometry change as "retreat" killed valid grasps 15
        # steps into lift.  The hard Cartesian workspace check below remains
        # active throughout the entire episode.
        # ``near`` may become false again after the gripper has caught the
        # sphere and carries it behind the fixed interception line.  Latch
        # the first near-field entry for the whole episode so grasp/lift can
        # never reactivate the coarse-stage retreat detector.
        self.near_latched |= near
        comparable &= ~self.near_latched
        comparable &= distance >= float(retreat_min_distance)
        self.retreat_count = update_retreat_count(
            self.retreat_count,
            self.previous_distance,
            distance,
            comparable,
            float(retreat_epsilon),
        )
        self.previous_distance.copy_(distance)
        self.previous_near.copy_(near)
        self.initialized.fill_(True)
        return outside_workspace(tcp, workspace_lower, workspace_upper) | (
            self.retreat_count >= int(retreat_steps)
        )


def asset_off_belt(
    env: ManagerBasedRLEnv,
    belt_y: float,
    belt_half_width: float,
    surface_z: float,
    z_margin: float = 0.02,
    lateral_check_max_z: float | None = None,
    lateral_safe_radius: float | None = None,
    lateral_safe_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("food"),
) -> torch.Tensor:
    """A moving task object fell below the belt surface or left it laterally.

    With ``lateral_check_max_z``, leaving sideways only counts near the belt.
    ``lateral_safe_radius`` additionally exempts the target receptacle's xy
    footprint, because a placed object eventually descends back to table
    height outside the belt. Falling below the belt remains a failure.
    """
    pos = _pos_cell(env, asset_cfg)
    lateral = (pos[:, 1] - belt_y).abs() > belt_half_width
    if lateral_check_max_z is not None:
        lateral &= pos[:, 2] <= lateral_check_max_z
    if lateral_safe_radius is not None:
        safe_pos = _pos_cell(env, lateral_safe_cfg)
        inside_safe_xy = torch.linalg.vector_norm(pos[:, :2] - safe_pos[:, :2], dim=-1) < lateral_safe_radius
        lateral &= ~inside_safe_xy
    return (pos[:, 2] < surface_z - z_margin) | lateral


def asset_lifted(
    env: ManagerBasedRLEnv,
    surface_z: float,
    item_radius: float,
    target_lift: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("food"),
) -> torch.Tensor:
    """C0 success: the grasped object is ``target_lift`` above its rest centre."""
    pos = _pos_cell(env, asset_cfg)
    held = grasped_mask(env, robot_cfg, asset_cfg, open_pos=open_pos, closed_pos=closed_pos)
    return held & (pos[:, 2] >= surface_z + item_radius + target_lift)


class contact_asset_lifted(ManagerTermBase):
    """Strict success: true 3 cm object lift while both fingers contact it."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.count.zero_()
        else:
            self.count[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        surface_z: float,
        item_radius: float,
        target_lift: float = 0.03,
        stable_steps: int = 2,
        force_threshold: float = 0.05,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_finger_contact"),
        right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_finger_contact"),
    ) -> torch.Tensor:
        pos = _pos_cell(env, asset_cfg)
        lifted = pos[:, 2] >= float(surface_z) + float(item_radius) + float(target_lift)
        held = dual_finger_contact_mask(env, left_sensor_cfg, right_sensor_cfg, force_threshold)
        valid = lifted & held
        self.count = torch.where(valid, self.count + 1, torch.zeros_like(self.count))
        return self.count >= int(stable_steps)


class coupled_contact_asset_lifted(ManagerTermBase):
    """Strict success after stable pre-lift contact and TCP-sphere coupling."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.hold_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.success_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.reference_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.coupled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.reference_relative = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.hold_count.zero_()
            self.success_count.zero_()
            self.reference_valid.zero_()
            self.coupled.zero_()
            self.reference_relative.zero_()
        else:
            self.hold_count[env_ids] = 0
            self.success_count[env_ids] = 0
            self.reference_valid[env_ids] = False
            self.coupled[env_ids] = False
            self.reference_relative[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        surface_z: float,
        item_radius: float,
        target_lift: float = 0.03,
        stable_hold_steps: int = 5,
        success_steps: int = 2,
        prelift_max_height: float = 0.005,
        max_relative_drift: float = 0.008,
        settled_reference: bool = False,
        force_threshold: float = 0.05,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_finger_contact"),
        right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_finger_contact"),
    ) -> torch.Tensor:
        if min(stable_hold_steps, success_steps) < 1:
            raise ValueError("stable_hold_steps and success_steps must be >= 1")
        food_w = env.scene[asset_cfg.name].data.root_pos_w.torch
        relative = food_w - env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :]
        pos = food_w - env.scene.env_origins
        lift = (pos[:, 2] - float(surface_z) - float(item_radius)).clamp_min(0.0)
        held = dual_finger_contact_mask(env, left_sensor_cfg, right_sensor_cfg, force_threshold)

        new_contact = held & ~self.reference_valid
        self.reference_relative[new_contact] = relative[new_contact]
        self.reference_valid |= new_contact
        drift = torch.linalg.vector_norm(relative - self.reference_relative, dim=-1)
        prelift_stable = (
            held
            & self.reference_valid
            & (lift <= float(prelift_max_height))
            & (drift <= float(max_relative_drift))
        )
        counting = prelift_stable & ~self.coupled
        self.hold_count = torch.where(
            counting,
            self.hold_count + 1,
            torch.where(self.coupled, self.hold_count, torch.zeros_like(self.hold_count)),
        )
        if settled_reference:
            # Measure post-grasp drift from the geometry at the end of the
            # required stable hold, not from the first transient touch.
            self.reference_relative[counting] = relative[counting]
        self.coupled |= self.hold_count >= int(stable_hold_steps)

        drift = torch.linalg.vector_norm(relative - self.reference_relative, dim=-1)
        coupling_valid = held & (drift <= float(max_relative_drift))
        lost = self.coupled & ~coupling_valid
        self.coupled[lost] = False
        self.reference_valid[~held | lost] = False
        self.hold_count[~held | lost] = 0
        valid_success = (
            self.coupled
            & coupling_valid
            & (lift >= float(target_lift))
        )
        self.success_count = torch.where(
            valid_success, self.success_count + 1, torch.zeros_like(self.success_count)
        )
        return self.success_count >= int(success_steps)


class geometric_asset_lifted(ManagerTermBase):
    """Kinematic lift success held for a few steps, without a grasp/contact flag."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.count.zero_()
        else:
            self.count[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        surface_z: float,
        item_radius: float,
        target_lift: float,
        max_tcp_distance: float,
        stable_steps: int,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        pos_w = env.scene[asset_cfg.name].data.root_pos_w.torch
        pos = pos_w - env.scene.env_origins
        tcp = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :]
        valid = (
            (pos[:, 2] >= surface_z + item_radius + target_lift)
            & (torch.linalg.vector_norm(pos_w - tcp, dim=-1) <= max_tcp_distance)
        )
        self.count = torch.where(valid, self.count + 1, torch.zeros_like(self.count))
        return self.count >= int(stable_steps)


class geometric_sustained_close(ManagerTermBase):
    """Successful aligned close held for several steps, using no velocity or contact signal.

    A valid blocked aperture is deliberately bounded on both sides: an open
    gripper has not grasped, while a fully closed gripper most likely missed
    the object.  The real-cell equivalent needs only object/TCP geometry and
    finger position encoders.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.count.zero_()
        else:
            self.count[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        distance_threshold: float,
        min_close_fraction: float,
        max_close_fraction: float,
        stable_steps: int,
        robot_cfg: SceneEntityCfg,
        open_pos: float,
        closed_pos: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        food = env.scene[asset_cfg.name].data.root_pos_w.torch
        tcp = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :]
        distance = torch.linalg.vector_norm(food - tcp, dim=-1)
        fingers = env.scene[robot_cfg.name].data.joint_pos.torch[:, robot_cfg.joint_ids]
        close_fraction = gripper_close_fraction(fingers, open_pos, closed_pos)
        valid = (
            (distance <= float(distance_threshold))
            & (close_fraction >= float(min_close_fraction))
            & (close_fraction <= float(max_close_fraction))
        )
        self.count = torch.where(valid, self.count + 1, torch.zeros_like(self.count))
        return self.count >= int(stable_steps)


class geometric_close_then_lift(ManagerTermBase):
    """Contact-free bridge success: stable close, then coupled micro-lift.

    The first phase latches only after a valid aperture remains aligned near
    the belt surface.  At that instant the food-to-TCP offset is recorded.
    Success then requires a small lift while that offset and aperture remain
    stable for several steps.  This deliberately uses only pose estimates and
    finger encoders, matching the intended real-robot observation contract.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.close_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.lift_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.ready_to_lift = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.reference_offset = torch.zeros(env.num_envs, 3, dtype=torch.float, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.close_count.zero_()
            self.lift_count.zero_()
            self.ready_to_lift.zero_()
            self.reference_offset.zero_()
        else:
            self.close_count[env_ids] = 0
            self.lift_count[env_ids] = 0
            self.ready_to_lift[env_ids] = False
            self.reference_offset[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        surface_z: float,
        item_radius: float,
        close_steps: int,
        distance_threshold: float,
        min_close_fraction: float,
        max_close_fraction: float,
        prelift_max_height: float,
        target_lift: float,
        max_relative_drift: float,
        success_steps: int,
        robot_cfg: SceneEntityCfg,
        open_pos: float,
        closed_pos: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        food_w = env.scene[asset_cfg.name].data.root_pos_w.torch
        food = food_w - env.scene.env_origins
        tcp_w = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :]
        offset = food_w - tcp_w
        distance = torch.linalg.vector_norm(offset, dim=-1)
        lift_height = (food[:, 2] - float(surface_z) - float(item_radius)).clamp_min(0.0)

        fingers = env.scene[robot_cfg.name].data.joint_pos.torch[:, robot_cfg.joint_ids]
        close_fraction = gripper_close_fraction(fingers, open_pos, closed_pos)
        aligned_close = (
            (distance <= float(distance_threshold))
            & (close_fraction >= float(min_close_fraction))
            & (close_fraction <= float(max_close_fraction))
        )

        acquiring = aligned_close & (lift_height <= float(prelift_max_height)) & ~self.ready_to_lift
        self.close_count = torch.where(
            self.ready_to_lift,
            self.close_count,
            torch.where(acquiring, self.close_count + 1, torch.zeros_like(self.close_count)),
        )
        newly_ready = ~self.ready_to_lift & (self.close_count >= int(close_steps))
        self.reference_offset = torch.where(newly_ready.unsqueeze(-1), offset, self.reference_offset)
        self.ready_to_lift |= newly_ready

        relative_drift = torch.linalg.vector_norm(offset - self.reference_offset, dim=-1)
        coupled = aligned_close & (relative_drift <= float(max_relative_drift))
        lost = self.ready_to_lift & ~coupled
        self.ready_to_lift &= ~lost
        self.close_count = torch.where(lost, torch.zeros_like(self.close_count), self.close_count)

        lifted = self.ready_to_lift & coupled & (lift_height >= float(target_lift))
        self.lift_count = torch.where(lifted, self.lift_count + 1, torch.zeros_like(self.lift_count))
        return self.lift_count >= int(success_steps)


class stable_grasp_asset_lifted(ManagerTermBase):
    """C0 success only after a near-surface grasp has first remained stable.

    Readiness is latched after ``stable_steps`` consecutive held steps no more
    than ``prelift_max_height`` above the object's rest centre.  Losing the
    object clears the latch.  This keeps the terminal +success bonus from
    bypassing the hold-then-lift curriculum.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.hold_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.ready_to_lift = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.hold_count.zero_()
            self.ready_to_lift.zero_()
        else:
            self.hold_count[env_ids] = 0
            self.ready_to_lift[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        surface_z: float,
        item_radius: float,
        target_lift: float,
        stable_steps: int,
        prelift_max_height: float,
        robot_cfg: SceneEntityCfg,
        open_pos: float,
        closed_pos: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ) -> torch.Tensor:
        pos = _pos_cell(env, asset_cfg)
        lift_height = (pos[:, 2] - surface_z - item_radius).clamp_min(0.0)
        held = grasped_mask(env, robot_cfg, asset_cfg, open_pos=open_pos, closed_pos=closed_pos)
        counting = held & (lift_height <= prelift_max_height) & ~self.ready_to_lift
        self.hold_count = torch.where(
            self.ready_to_lift,
            self.hold_count,
            torch.where(counting, self.hold_count + 1, torch.zeros_like(self.hold_count)),
        )
        self.ready_to_lift |= self.hold_count >= stable_steps
        self.ready_to_lift &= held
        return self.ready_to_lift & (lift_height >= target_lift)


def bowl_off_belt(
    env: ManagerBasedRLEnv,
    belt_y: float,
    belt_half_width: float,
    surface_z: float,
    z_margin: float = 0.02,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
):
    """The bowl dropped below the belt surface or left the belt laterally."""
    pos = _pos_cell(env, bowl_cfg)
    return (pos[:, 2] < surface_z - z_margin) | ((pos[:, 1] - belt_y).abs() > belt_half_width)


def bowl_tipped(env: ManagerBasedRLEnv, max_tilt_rad: float, bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl")):
    """The bowl's up-axis tilted more than ``max_tilt_rad`` from vertical."""
    q = env.scene[bowl_cfg.name].data.root_quat_w.torch  # (x, y, z, w)
    up_z = 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)
    return up_z < torch.cos(torch.tensor(max_tilt_rad, device=q.device))


def food_off_table(env: ManagerBasedRLEnv, minimum_height: float = -0.05, food_cfg: SceneEntityCfg = SceneEntityCfg("food")):
    """The food fell below the table surface."""
    return _pos_cell(env, food_cfg)[:, 2] < minimum_height


def released_in_bowl_mask(
    env: ManagerBasedRLEnv,
    inner_radius: float,
    base_thickness: float,
    rim_height: float,
    item_radius: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    food_cfg: SceneEntityCfg,
    bowl_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """The food is inside the bowl (radially within the inner wall, between base and rim) and not grasped.

    The single definition of "in the bowl" shared by the ``food_in_bowl`` success termination and the
    ``released_in_bowl`` reward, so the two can never disagree.
    """
    food_pos, bowl_pos = env.scene[food_cfg.name].data.root_pos_w.torch, env.scene[bowl_cfg.name].data.root_pos_w.torch
    rel = food_pos - bowl_pos  # bowl origin = bottom center
    radial_ok = torch.linalg.vector_norm(rel[:, :2], dim=-1) < inner_radius - 0.5 * item_radius
    height_ok = (rel[:, 2] > base_thickness) & (rel[:, 2] < base_thickness + rim_height)
    released = ~grasped_mask(env, robot_cfg, food_cfg, open_pos=open_pos, closed_pos=closed_pos)
    return radial_ok & height_ok & released


def tcp_home_distance(
    env: ManagerBasedRLEnv,
    home_pos: tuple[float, float, float],
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Distance [m] of the TCP (the ``ee_frame`` target the other reward terms use, env-local) from ``home_pos``,
    the TCP position at the arm's default joint pose (``ArmCfg.home_tcp_pos``, measured once).

    "Home" is defined in end-effector position space, not joint space: the policy acts on the end-effector pose
    (IK), which leaves the Franka's redundant elbow free to drift, so a joint-space home could be unreachable.
    Shared by the success check and the ``return_home`` reward, so the two always agree.
    """
    tcp = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :] - env.scene.env_origins
    return torch.linalg.vector_norm(tcp - torch.tensor(home_pos, device=tcp.device), dim=-1)


REL_SPEED_THRESHOLD = 0.05
"""Food-to-bowl speed [m/s] below which the food counts as at rest in the bowl (``settled_in_bowl_mask``)."""


def settled_in_bowl_mask(
    env: ManagerBasedRLEnv,
    inner_radius: float,
    base_thickness: float,
    rim_height: float,
    item_radius: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    food_cfg: SceneEntityCfg,
    bowl_cfg: SceneEntityCfg,
    rel_speed_threshold: float = REL_SPEED_THRESHOLD,
) -> torch.Tensor:
    """``released_in_bowl_mask`` and moving slower than ``rel_speed_threshold`` relative to the bowl.

    One step of the ``food_in_bowl`` success condition (which additionally requires it for ``settle_steps``
    consecutive steps); the continuous demo counts placements with the same function.
    """
    food, bowl = env.scene[food_cfg.name].data, env.scene[bowl_cfg.name].data
    in_bowl = released_in_bowl_mask(
        env, inner_radius, base_thickness, rim_height, item_radius, robot_cfg, open_pos, closed_pos,
        food_cfg, bowl_cfg,
    )
    rel_speed = torch.linalg.vector_norm(food.root_lin_vel_w.torch - bowl.root_lin_vel_w.torch, dim=-1)
    return in_bowl & (rel_speed < rel_speed_threshold)


def never(env: ManagerBasedRLEnv) -> torch.Tensor:
    """A termination that never fires (the continuous demo disables every episode end with it)."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


class food_in_bowl(ManagerTermBase):
    """Success: food inside the target bowl, released, and at rest relative to the bowl for ``settle_steps``.

    With ``home_tolerance`` set (env option ``success_requires_home``) it additionally requires the TCP within
    ``home_tolerance`` [m] of its home position (``tcp_home_distance``): the episode ends only once the
    food is placed AND the arm is back home. The settle counter keeps counting meanwhile, so success fires on the
    first step the arm is home after the food has settled. Unset (the default), nothing changes."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.counter.zero_()
        else:
            self.counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        inner_radius: float,
        base_thickness: float,
        rim_height: float,
        item_radius: float,
        settle_steps: int,
        robot_cfg: SceneEntityCfg,
        open_pos: float,
        closed_pos: float,
        rel_speed_threshold: float = REL_SPEED_THRESHOLD,
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
        home_tolerance: float | None = None,
        home_pos: tuple[float, float, float] | None = None,
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        inside = settled_in_bowl_mask(
            env, inner_radius, base_thickness, rim_height, item_radius, robot_cfg, open_pos, closed_pos,
            food_cfg, bowl_cfg, rel_speed_threshold,
        )
        self.counter = torch.where(inside, self.counter + 1, torch.zeros_like(self.counter))
        placed = self.counter >= settle_steps
        if home_tolerance is None:
            return placed
        return placed & (tcp_home_distance(env, home_pos, ee_frame_cfg) <= home_tolerance)
