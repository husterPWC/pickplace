"""Reward terms (Isaac Lab scales each term by step_dt)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from pickplace.envs.mdp.observations import (
    dual_finger_contact_mask,
    gripper_blocked_between_limits,
    gripper_close_fraction,
    grasped_mask,
    sphere_best_grasp_pose_delta,
)
from pickplace.envs.mdp.terminations import released_in_bowl_mask, tcp_home_distance
from pickplace.intercept import (
    interception_objective,
    multiscale_position_quality,
    progress_augmented_step_reward,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def zero(env: ManagerBasedRLEnv) -> torch.Tensor:
    """A shape-correct inactive dense term retained for the shared reward vector contract."""
    return torch.zeros(env.num_envs, device=env.device)


def _tcp_w(env, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :]


def reach_food(
    env: ManagerBasedRLEnv,
    std: float,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(env.scene[food_cfg.name].data.root_pos_w.torch - _tcp_w(env, ee_frame_cfg), dim=-1)
    return 1.0 - torch.tanh(distance / std)


def sphere_grasp_pose_approach(
    env: ManagerBasedRLEnv,
    translation_std: float = 0.05,
    rotation_std: float = 0.35,
    num_candidates: int = 40,
    tcp_height_offset: float = 0.002,
    canonical_yaw: float | None = None,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
) -> torch.Tensor:
    """Dense quality of the nearest valid 6D grasp, not object-centre reach."""
    delta = sphere_best_grasp_pose_delta(
        env,
        num_candidates=num_candidates,
        tcp_height_offset=tcp_height_offset,
        canonical_yaw=canonical_yaw,
        food_cfg=food_cfg,
    )
    positional = torch.linalg.vector_norm(delta[:, :3], dim=-1) / float(translation_std)
    angular = torch.linalg.vector_norm(delta[:, 3:], dim=-1) / float(rotation_std)
    return 1.0 - torch.tanh(positional + angular)


def multiscale_pose_quality(
    translation_error: torch.Tensor,
    rotation_error: torch.Tensor,
    translation_near_scale: float = 0.05,
    translation_far_scale: float = 0.35,
    rotation_near_scale: float = 0.35,
    rotation_far_scale: float = 1.5,
    near_weight: float = 0.3,
) -> torch.Tensor:
    """Bounded pose quality with useful slope both far away and near a grasp.

    The original single 5 cm tanh is appropriate for fine alignment, but it
    saturates to zero at the 40--50 cm reset distances used by this task.  A
    broad component supplies a learnable global approach signal while the
    narrow component retains centimetre/radian precision.  Inputs are norms,
    which keeps this helper simulator-free and directly testable as a reward
    landscape gate.
    """
    scales = (
        translation_near_scale,
        translation_far_scale,
        rotation_near_scale,
        rotation_far_scale,
    )
    if any(float(scale) <= 0.0 for scale in scales):
        raise ValueError("multiscale pose-quality scales must be > 0")
    if not 0.0 <= float(near_weight) <= 1.0:
        raise ValueError("near_weight must be in [0, 1]")
    near = 1.0 - torch.tanh(
        translation_error / float(translation_near_scale)
        + rotation_error / float(rotation_near_scale)
    )
    far = 1.0 - torch.tanh(
        translation_error / float(translation_far_scale)
        + rotation_error / float(rotation_far_scale)
    )
    return float(near_weight) * near + (1.0 - float(near_weight)) * far


def sphere_intercept_pose_approach(
    env: ManagerBasedRLEnv,
    intercept_x: float = 0.45,
    intercept_lead_distance: float = 0.08,
    hover_height: float = 0.08,
    transition_width: float = 0.01,
    num_candidates: int = 40,
    tcp_height_offset: float = 0.002,
    canonical_yaw: float | None = None,
    trigger_translation: float | None = None,
    rotation_threshold: float | None = None,
    trigger_bonus: float = 0.0,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Position-only staged shaping for a fixed conveyor interception line.

    Upstream of the line, reward the TCP for waiting above ``intercept_x`` at
    the sphere's current lateral coordinate.  As the sphere reaches the lead
    window, blend continuously into nearest-6D grasp-pose quality.  This uses
    current geometry only: neither belt nor object velocity is read.
    """
    if min(intercept_lead_distance, hover_height, transition_width) <= 0.0:
        raise ValueError("intercept lead, hover height, and transition width must be positive")
    delta = sphere_best_grasp_pose_delta(
        env,
        num_candidates=num_candidates,
        tcp_height_offset=tcp_height_offset,
        canonical_yaw=canonical_yaw,
        food_cfg=food_cfg,
        ee_frame_cfg=ee_frame_cfg,
    )
    tcp_cell = _tcp_w(env, ee_frame_cfg) - env.scene.env_origins
    target_x = tcp_cell[:, 0] + delta[:, 0]
    hover_delta = delta[:, :3].clone()
    hover_delta[:, 0] = float(intercept_x) - tcp_cell[:, 0]
    hover_delta[:, 2] += float(hover_height)
    rotation_error = torch.linalg.vector_norm(delta[:, 3:], dim=-1)
    hover_quality = multiscale_pose_quality(
        torch.linalg.vector_norm(hover_delta, dim=-1), rotation_error
    )
    grasp_quality = multiscale_pose_quality(
        torch.linalg.vector_norm(delta[:, :3], dim=-1), rotation_error
    )
    trigger_x = float(intercept_x) - float(intercept_lead_distance)
    arrival = torch.sigmoid((target_x - trigger_x) / float(transition_width))
    quality = (1.0 - arrival) * hover_quality + arrival * grasp_quality
    if trigger_bonus:
        if trigger_translation is None or rotation_threshold is None:
            raise ValueError("trigger bonus requires trigger_translation and rotation_threshold")
        if min(float(trigger_translation), float(rotation_threshold), float(trigger_bonus)) <= 0.0:
            raise ValueError("trigger thresholds and bonus must be positive")
        translation_error = torch.linalg.vector_norm(delta[:, :3], dim=-1)
        # Smooth approximations of the exact option gates preserve a useful
        # gradient on both sides of the boundary.  Repeated reward inside the
        # basin also teaches the policy to remain aligned for consecutive
        # steps, while the option retains final authority over triggering.
        translation_gate = torch.sigmoid(
            (float(trigger_translation) - translation_error) / max(0.25 * float(trigger_translation), 1.0e-6)
        )
        rotation_gate = torch.sigmoid(
            (float(rotation_threshold) - rotation_error) / max(0.25 * float(rotation_threshold), 1.0e-6)
        )
        quality = quality + arrival * float(trigger_bonus) * translation_gate * rotation_gate
    return quality


class sphere_residual_intercept_approach(ManagerTermBase):
    """Fixed-hover/near-grasp quality plus an explicitly signed progress term.

    Far from arrival, absolute quality depends *only* on position at the fixed
    interception hover point.  Near the line, 6D grasp-pose quality becomes
    active.  Isaac multiplies dense rewards by ``step_dt``; returning
    ``progress_gain * (d_prev-d_now) / step_dt`` therefore contributes exactly
    ``progress_gain * (d_prev-d_now)`` to the unweighted per-step return.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.previous_distance = torch.zeros(env.num_envs, device=env.device)
        self.previous_near = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.previous_distance.zero_()
            self.previous_near.zero_()
            self.initialized.zero_()
        else:
            self.previous_distance[env_ids] = 0.0
            self.previous_near[env_ids] = False
            self.initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        intercept_x: float = 0.45,
        intercept_lead_distance: float = 0.08,
        intercept_align_lead_distance: float = 0.04,
        hover_height: float = 0.08,
        pre_descent_height: float | None = None,
        progress_gain: float = 20.0,
        num_candidates: int = 40,
        tcp_height_offset: float = 0.002,
        canonical_yaw: float | None = 0.0,
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if min(intercept_lead_distance, hover_height, progress_gain) <= 0.0:
            raise ValueError("intercept distances and progress_gain must be positive")
        pose_delta = sphere_best_grasp_pose_delta(
            env,
            num_candidates=num_candidates,
            tcp_height_offset=tcp_height_offset,
            canonical_yaw=canonical_yaw,
            food_cfg=food_cfg,
            ee_frame_cfg=ee_frame_cfg,
        )
        tcp_cell = _tcp_w(env, ee_frame_cfg) - env.scene.env_origins
        distance, near, _ = interception_objective(
            tcp_cell,
            pose_delta,
            intercept_x,
            intercept_lead_distance,
            hover_height,
            intercept_align_lead_distance,
            pre_descent_height,
        )
        rotation_error = torch.linalg.vector_norm(pose_delta[:, 3:], dim=-1)
        far_quality = multiscale_position_quality(distance)
        near_quality = multiscale_pose_quality(distance, rotation_error)
        quality = torch.where(near, near_quality, far_quality)

        comparable = self.initialized & near.eq(self.previous_near)
        result = progress_augmented_step_reward(
            quality,
            self.previous_distance,
            distance,
            float(env.step_dt),
            float(progress_gain),
            comparable,
        )
        self.previous_distance.copy_(distance)
        self.previous_near.copy_(near)
        self.initialized.fill_(True)
        return result


def sphere_grasp_pose_approach_multiscale(
    env: ManagerBasedRLEnv,
    translation_near_scale: float = 0.05,
    translation_far_scale: float = 0.35,
    rotation_near_scale: float = 0.35,
    rotation_far_scale: float = 1.5,
    near_weight: float = 0.3,
    num_candidates: int = 40,
    tcp_height_offset: float = 0.002,
    canonical_yaw: float | None = None,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
) -> torch.Tensor:
    """Nearest valid 6D sphere-grasp quality without the old far-field dead zone."""
    delta = sphere_best_grasp_pose_delta(
        env,
        num_candidates=num_candidates,
        tcp_height_offset=tcp_height_offset,
        canonical_yaw=canonical_yaw,
        food_cfg=food_cfg,
    )
    return multiscale_pose_quality(
        torch.linalg.vector_norm(delta[:, :3], dim=-1),
        torch.linalg.vector_norm(delta[:, 3:], dim=-1),
        translation_near_scale=translation_near_scale,
        translation_far_scale=translation_far_scale,
        rotation_near_scale=rotation_near_scale,
        rotation_far_scale=rotation_far_scale,
        near_weight=near_weight,
    )


def dual_finger_grasp(
    env: ManagerBasedRLEnv,
    force_threshold: float = 0.05,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_finger_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_finger_contact"),
) -> torch.Tensor:
    """Hidden simulator reward label; never exposed in actor observations."""
    return dual_finger_contact_mask(env, left_sensor_cfg, right_sensor_cfg, force_threshold).float()


def contact_lift_progress(
    env: ManagerBasedRLEnv,
    surface_z: float,
    item_radius: float,
    target_lift: float,
    force_threshold: float = 0.05,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_finger_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_finger_contact"),
) -> torch.Tensor:
    """0..1 physical lift progress, gated by simultaneous two-finger contact."""
    z = env.scene[food_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    progress = ((z - float(surface_z) - float(item_radius)) / float(target_lift)).clamp(0.0, 1.0)
    held = dual_finger_contact_mask(env, left_sensor_cfg, right_sensor_cfg, force_threshold)
    return held.float() * progress


class coupled_contact_lift_progress(ManagerTermBase):
    """True sphere lift progress after a stable, geometrically coupled hold.

    Finger contact is a simulator-only training label.  It first has to remain
    bilateral near belt height while the sphere-to-TCP vector changes by less
    than ``max_relative_drift``.  Only then can actual sphere-height progress
    earn reward.  A lost contact or excessive relative drift immediately
    removes the lift reward, closing both the aperture and collision/bounce
    false-positive paths.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.hold_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.reference_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.coupled = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.reference_relative = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.hold_count.zero_()
            self.reference_valid.zero_()
            self.coupled.zero_()
            self.reference_relative.zero_()
        else:
            self.hold_count[env_ids] = 0
            self.reference_valid[env_ids] = False
            self.coupled[env_ids] = False
            self.reference_relative[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        surface_z: float,
        item_radius: float,
        target_lift: float,
        stable_hold_steps: int = 5,
        prelift_max_height: float = 0.005,
        max_relative_drift: float = 0.008,
        settled_reference: bool = False,
        force_threshold: float = 0.05,
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_finger_contact"),
        right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_finger_contact"),
    ) -> torch.Tensor:
        if stable_hold_steps < 1 or min(prelift_max_height, max_relative_drift, target_lift) <= 0.0:
            raise ValueError("coupled lift thresholds and stable_hold_steps must be positive")
        food_w = env.scene[food_cfg.name].data.root_pos_w.torch
        relative = food_w - _tcp_w(env, ee_frame_cfg)
        z = food_w[:, 2] - env.scene.env_origins[:, 2]
        lift = (z - float(surface_z) - float(item_radius)).clamp_min(0.0)
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
            # During the stable pre-lift hold the sphere is still
            # self-centering between the fingers, especially at conveyor
            # speed.  Latch the final settled geometry, while still requiring
            # every inter-step change to remain inside max_relative_drift.
            self.reference_relative[counting] = relative[counting]
        self.coupled |= self.hold_count >= int(stable_hold_steps)

        drift = torch.linalg.vector_norm(relative - self.reference_relative, dim=-1)
        coupling_valid = held & (drift <= float(max_relative_drift))
        lost = self.coupled & ~coupling_valid
        self.coupled[lost] = False
        self.reference_valid[~held | lost] = False
        self.hold_count[~held | lost] = 0

        progress = (lift / float(target_lift)).clamp(0.0, 1.0)
        return (self.coupled & coupling_valid).float() * progress


def grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
) -> torch.Tensor:
    """Fingers closed on the food near the TCP, no lift required: the stepping stone between reach_food and grasp_lift."""
    return grasped_mask(env, robot_cfg, food_cfg, open_pos=open_pos, closed_pos=closed_pos).float()


def grasp_lift(
    env: ManagerBasedRLEnv,
    lift_height: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
) -> torch.Tensor:
    food_z = env.scene[food_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    grasped = grasped_mask(env, robot_cfg, food_cfg, open_pos=open_pos, closed_pos=closed_pos)
    return (grasped & (food_z > lift_height)).float()


def grasp_lift_progress(
    env: ManagerBasedRLEnv,
    surface_z: float,
    item_radius: float,
    target_lift: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
) -> torch.Tensor:
    """Continuous 0..1 held-lift progress for the C0 curriculum.

    Unlike the ordinary binary ``grasp_lift``, every small upward move is
    informative. The grasp gate prevents bouncing or rolling an unheld cube
    from earning lift reward.
    """
    food_z = env.scene[food_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    rest_center_z = surface_z + item_radius
    progress = ((food_z - rest_center_z) / target_lift).clamp(0.0, 1.0)
    held = grasped_mask(env, robot_cfg, food_cfg, open_pos=open_pos, closed_pos=closed_pos)
    return held.float() * progress


def geometric_close_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    far_close_penalty: float = 0.0,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closing only while geometrically aligned with the object.

    This deliberately does not call :func:`grasped_mask`: finger encoders,
    object/TCP pose and nothing contact-derived define the signal.
    """
    distance = torch.linalg.vector_norm(
        env.scene[food_cfg.name].data.root_pos_w.torch - _tcp_w(env, ee_frame_cfg), dim=-1
    )
    fingers = env.scene[robot_cfg.name].data.joint_pos.torch[:, robot_cfg.joint_ids]
    close_fraction = gripper_close_fraction(fingers, open_pos, closed_pos)
    proximity = 1.0 - torch.tanh(distance / std)
    # Stage-one curricula need an explicit reason to stay open while chasing
    # the cube; otherwise "always closed" earns the same zero far away and can
    # become a dead-end policy.  Zero preserves every historical run.
    return close_fraction * (proximity - float(far_close_penalty) * (1.0 - proximity))


class geometric_lift_progress(ManagerTermBase):
    """Lift/coupling reward and one-shot slip penalty from kinematics only."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.peak_lift = torch.zeros(env.num_envs, device=env.device)
        self.slip_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.previous_ee = torch.zeros(env.num_envs, 3, device=env.device)
        self.motion_initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.peak_lift.zero_()
            self.slip_latched.zero_()
            self.previous_ee.zero_()
            self.motion_initialized.zero_()
        else:
            self.peak_lift[env_ids] = 0.0
            self.slip_latched[env_ids] = False
            self.previous_ee[env_ids] = 0.0
            self.motion_initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        surface_z: float,
        item_radius: float,
        target_lift: float,
        coupling_std: float,
        slip_min_lift: float,
        slip_distance: float,
        slip_fall: float,
        slip_penalty: float,
        relative_speed_weight: float = 0.0,
        relative_speed_std: float = 0.05,
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        food_pos = env.scene[food_cfg.name].data.root_pos_w.torch
        food_z = food_pos[:, 2] - env.scene.env_origins[:, 2]
        lift = (food_z - surface_z - item_radius).clamp_min(0.0)
        distance = torch.linalg.vector_norm(food_pos - _tcp_w(env, ee_frame_cfg), dim=-1)
        progress = (lift / target_lift).clamp(0.0, 1.0)
        coupling = 1.0 - torch.tanh(distance / coupling_std)

        # Relative velocity is an earlier indication of incipient slip than
        # position separation.  Blend it as a positive multiplier only: this
        # cannot create a new negative-reward avoidance strategy, and weight=0
        # preserves the historical C0m reward exactly.
        if relative_speed_weight > 0:
            ee_velocity = (_tcp_w(env, ee_frame_cfg) - self.previous_ee) / float(env.step_dt)
            relative_speed = torch.linalg.vector_norm(
                env.scene[food_cfg.name].data.root_lin_vel_w.torch - ee_velocity, dim=-1
            )
            velocity_quality = 1.0 - torch.tanh(relative_speed / relative_speed_std)
            velocity_quality = torch.where(
                self.motion_initialized, velocity_quality, torch.ones_like(velocity_quality)
            )
            coupling *= 1.0 - relative_speed_weight * (1.0 - velocity_quality)

        previous_peak = self.peak_lift.clone()
        self.peak_lift = torch.maximum(self.peak_lift, lift)
        slipped_now = (
            (previous_peak >= slip_min_lift)
            & ((distance > slip_distance) | (lift < previous_peak - slip_fall))
            & ~self.slip_latched
        )
        reward = progress * coupling
        if slip_penalty > 0:
            reward -= slipped_now.float() * slip_penalty / env.step_dt
        self.slip_latched |= slipped_now
        # A clearly reset-to-belt, re-aligned state permits one later retry.
        retry = (lift < 0.003) & (distance < slip_distance)
        self.slip_latched &= ~retry
        self.peak_lift = torch.where(retry, lift, self.peak_lift)
        self.previous_ee.copy_(_tcp_w(env, ee_frame_cfg))
        self.motion_initialized.fill_(True)
        return reward


class geometric_bridge_lift_progress(ManagerTermBase):
    """Reward only post-close, geometrically coupled upward progress.

    A small TCP-progress component gives the old close-only policy an immediate
    reason to reverse its learned downward z action.  Object progress remains
    dominant, and both components vanish as the latched food/TCP offset drifts
    to the bridge limit.  No velocity, contact, force, tactile, or current
    signal is used.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n = env.num_envs
        self.close_count = torch.zeros(n, dtype=torch.long, device=env.device)
        self.latched = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.reference_offset = torch.zeros(n, 3, device=env.device)
        self.reference_tcp_z = torch.zeros(n, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.close_count.zero_()
            self.latched.zero_()
            self.reference_offset.zero_()
            self.reference_tcp_z.zero_()
        else:
            self.close_count[env_ids] = 0
            self.latched[env_ids] = False
            self.reference_offset[env_ids] = 0.0
            self.reference_tcp_z[env_ids] = 0.0

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
        tcp_progress_weight: float,
        object_progress_weight: float,
        robot_cfg: SceneEntityCfg,
        open_pos: float,
        closed_pos: float,
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        food_w = env.scene[food_cfg.name].data.root_pos_w.torch
        food = food_w - env.scene.env_origins
        tcp_w = _tcp_w(env, ee_frame_cfg)
        offset = food_w - tcp_w
        distance = torch.linalg.vector_norm(offset, dim=-1)
        object_lift = (food[:, 2] - float(surface_z) - float(item_radius)).clamp_min(0.0)

        fingers = env.scene[robot_cfg.name].data.joint_pos.torch[:, robot_cfg.joint_ids]
        close_fraction = gripper_close_fraction(fingers, open_pos, closed_pos)
        aligned_close = (
            (distance <= float(distance_threshold))
            & (close_fraction >= float(min_close_fraction))
            & (close_fraction <= float(max_close_fraction))
        )
        acquiring = aligned_close & (object_lift <= float(prelift_max_height)) & ~self.latched
        self.close_count = torch.where(
            self.latched,
            self.close_count,
            torch.where(acquiring, self.close_count + 1, torch.zeros_like(self.close_count)),
        )
        newly_latched = ~self.latched & (self.close_count >= int(close_steps))
        self.reference_offset = torch.where(newly_latched[:, None], offset, self.reference_offset)
        self.reference_tcp_z = torch.where(newly_latched, tcp_w[:, 2], self.reference_tcp_z)
        self.latched |= newly_latched

        drift = torch.linalg.vector_norm(offset - self.reference_offset, dim=-1)
        coupled = aligned_close & (drift <= float(max_relative_drift))
        coupling_quality = (1.0 - drift / float(max_relative_drift)).clamp(0.0, 1.0)
        tcp_progress = ((tcp_w[:, 2] - self.reference_tcp_z) / float(target_lift)).clamp(0.0, 1.0)
        object_progress = (object_lift / float(target_lift)).clamp(0.0, 1.0)
        reward = coupling_quality * (
            float(tcp_progress_weight) * tcp_progress
            + float(object_progress_weight) * object_progress
        )
        reward = torch.where(self.latched & coupled, reward, torch.zeros_like(reward))

        lost = self.latched & ~coupled
        self.latched &= ~lost
        self.close_count = torch.where(lost, torch.zeros_like(self.close_count), self.close_count)
        return reward


class geometric_stable_close(ManagerTermBase):
    """Reward a temporally stable, contact-free close and penalise confidence loss.

    This is a lightweight grasp-target stabiliser for moving objects.  It uses
    only object/TCP pose, their finite-difference relative velocity, and finger
    aperture.  Those quantities are available from cameras, robot FK, and the
    gripper encoder on the real cell; no simulator contact, force, tactile,
    torque, or motor-current signal is queried.

    A grasp target must stay close and velocity-matched for ``stable_steps``
    before full closing reward is available.  Once the fingers are geometrically
    coupled for the same duration, losing that confidence emits a one-shot
    penalty.  Hysteresis prevents single-frame vision/control noise from
    repeatedly toggling either latch.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n = env.num_envs
        self.ready_count = torch.zeros(n, dtype=torch.long, device=env.device)
        self.coupled_count = torch.zeros(n, dtype=torch.long, device=env.device)
        self.target_latched = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.confidence_latched = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.previous_ee = torch.zeros(n, 3, device=env.device)
        self.previous_aperture = torch.zeros(n, device=env.device)
        self.initialized = torch.zeros(n, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.ready_count.zero_()
            self.coupled_count.zero_()
            self.target_latched.zero_()
            self.confidence_latched.zero_()
            self.previous_ee.zero_()
            self.previous_aperture.zero_()
            self.initialized.zero_()
        else:
            self.ready_count[env_ids] = 0
            self.coupled_count[env_ids] = 0
            self.target_latched[env_ids] = False
            self.confidence_latched[env_ids] = False
            self.previous_ee[env_ids] = 0.0
            self.previous_aperture[env_ids] = 0.0
            self.initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        stable_steps: int,
        distance_threshold: float,
        relative_speed_threshold: float,
        distance_std: float,
        relative_speed_std: float,
        hold_reward: float,
        premature_close_penalty: float,
        confidence_loss_penalty: float,
        robot_cfg: SceneEntityCfg,
        open_pos: float,
        closed_pos: float,
        margin: float = 0.004,
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if stable_steps < 1:
            raise ValueError(f"stable_steps must be >= 1, got {stable_steps}")

        food = env.scene[food_cfg.name].data.root_pos_w.torch
        food_velocity = env.scene[food_cfg.name].data.root_lin_vel_w.torch
        ee = _tcp_w(env, ee_frame_cfg)
        fingers = env.scene[robot_cfg.name].data.joint_pos.torch[:, robot_cfg.joint_ids]
        aperture = fingers.mean(dim=-1)
        distance = torch.linalg.vector_norm(food - ee, dim=-1)
        ee_velocity = (ee - self.previous_ee) / float(env.step_dt)
        relative_speed = torch.linalg.vector_norm(food_velocity - ee_velocity, dim=-1)
        # The first call after reset has no finite-difference TCP velocity.
        # Marking it unready avoids manufacturing a zero-relative-speed frame.
        valid_motion = self.initialized

        aligned = distance < distance_threshold
        velocity_matched = relative_speed < relative_speed_threshold
        ready = valid_motion & aligned & velocity_matched
        self.ready_count = torch.where(ready, self.ready_count + 1, torch.zeros_like(self.ready_count))
        self.target_latched |= self.ready_count >= stable_steps
        target_hysteresis = (
            (distance < 1.5 * distance_threshold)
            & (relative_speed < 2.0 * relative_speed_threshold)
        )
        self.target_latched &= target_hysteresis | ~valid_motion

        blocked = gripper_blocked_between_limits(fingers, open_pos, closed_pos, margin)
        coupled = self.target_latched & aligned & velocity_matched & blocked
        self.coupled_count = torch.where(
            coupled, self.coupled_count + 1, torch.zeros_like(self.coupled_count)
        )
        newly_confident = self.coupled_count >= stable_steps
        self.confidence_latched |= newly_confident
        confidence_hysteresis = (
            blocked
            & (distance < 1.5 * distance_threshold)
            & (relative_speed < 2.0 * relative_speed_threshold)
        )
        confidence_lost = self.confidence_latched & ~confidence_hysteresis
        self.confidence_latched &= ~confidence_lost

        close_fraction = gripper_close_fraction(fingers, open_pos, closed_pos)
        proximity = 1.0 - torch.tanh(distance / distance_std)
        velocity_quality = 1.0 - torch.tanh(relative_speed / relative_speed_std)
        ready_fraction = (self.ready_count.float() / stable_steps).clamp(0.0, 1.0)
        coupled_fraction = (self.coupled_count.float() / stable_steps).clamp(0.0, 1.0)

        # Penalise only actual closing motion before stability, rather than a
        # closed gripper on every later frame.  This keeps the shaping local.
        closing_direction = 1.0 if float(closed_pos) > float(open_pos) else -1.0
        closing_now = valid_motion & (
            closing_direction * (aperture - self.previous_aperture) > 1.0e-4
        )
        reward = close_fraction * proximity * velocity_quality * ready_fraction
        reward += hold_reward * coupled_fraction * proximity * velocity_quality
        reward -= closing_now.float() * (1.0 - ready_fraction) * premature_close_penalty
        if confidence_loss_penalty > 0:
            reward -= confidence_lost.float() * confidence_loss_penalty / float(env.step_dt)

        self.previous_ee.copy_(ee)
        self.previous_aperture.copy_(aperture)
        self.initialized.fill_(True)
        return reward


class stable_grasp_stage(ManagerTermBase):
    """Reward consecutive grasp stability and penalise the instant a grasp slips.

    ``hold_reward`` is a dense per-second reward. ``slip_penalty`` is a
    one-shot return penalty: division by ``step_dt`` cancels Isaac Lab's dense
    reward integration. A slip is the transition held -> not held, so a cube
    that was merely approached or missed is never penalised.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.hold_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.was_held = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.hold_count.zero_()
            self.was_held.zero_()
        else:
            self.hold_count[env_ids] = 0
            self.was_held[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        stable_steps: int,
        hold_reward: float,
        slip_penalty: float,
        robot_cfg: SceneEntityCfg,
        open_pos: float,
        closed_pos: float,
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ) -> torch.Tensor:
        if stable_steps < 1:
            raise ValueError(f"stable_steps must be >= 1, got {stable_steps}")
        held = grasped_mask(env, robot_cfg, food_cfg, open_pos=open_pos, closed_pos=closed_pos)
        slipped = self.was_held & ~held
        self.hold_count = torch.where(held, self.hold_count + 1, torch.zeros_like(self.hold_count))
        progress = (self.hold_count.float() / stable_steps).clamp(max=1.0)
        reward = held.float() * progress * hold_reward
        reward -= slipped.float() * slip_penalty / env.step_dt
        self.was_held.copy_(held)
        return reward


class stable_grasp_lift_progress(ManagerTermBase):
    """Held-lift progress gated until the pre-lift grasp is stable.

    With ``prelift_max_height`` unset, this is the original consecutive-hold
    gate.  When it is set, the counter advances only while the object remains
    close to its resting height and latches once stable.  This implements an
    actual hold-then-lift state machine instead of allowing an immediate lift
    to satisfy the hold counter while already moving upward.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
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
        stable_steps: int,
        surface_z: float,
        item_radius: float,
        target_lift: float,
        prelift_max_height: float | None,
        premature_lift_penalty: float,
        robot_cfg: SceneEntityCfg,
        open_pos: float,
        closed_pos: float,
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ) -> torch.Tensor:
        if stable_steps < 1:
            raise ValueError(f"stable_steps must be >= 1, got {stable_steps}")
        held = grasped_mask(env, robot_cfg, food_cfg, open_pos=open_pos, closed_pos=closed_pos)
        food_z = env.scene[food_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
        rest_center_z = surface_z + item_radius
        lift_height = (food_z - rest_center_z).clamp_min(0.0)
        lift_progress = (lift_height / target_lift).clamp(0.0, 1.0)
        prelift_ok = held
        if prelift_max_height is not None:
            prelift_ok = prelift_ok & (lift_height <= prelift_max_height)
        counting = prelift_ok & ~self.ready_to_lift
        self.hold_count = torch.where(
            self.ready_to_lift,
            self.hold_count,
            torch.where(counting, self.hold_count + 1, torch.zeros_like(self.hold_count)),
        )
        self.ready_to_lift |= self.hold_count >= stable_steps
        self.ready_to_lift &= held

        reward = (held & self.ready_to_lift).float() * lift_progress
        if prelift_max_height is not None and premature_lift_penalty > 0:
            premature_progress = ((lift_height - prelift_max_height) / target_lift).clamp(0.0, 1.0)
            premature = held & ~self.ready_to_lift
            reward -= premature.float() * premature_progress * premature_lift_penalty
        return reward


def transport_to_bowl(
    env: ManagerBasedRLEnv,
    std: float,
    hover_height: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    lift_height: float,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """While the food is actually held (grasped *and* lifted above ``lift_height``), approach a point above the
    bowl's *current* position (a moving target).

    Requiring the lift (not just ``grasped_mask``) closes the exploit where a policy shepherds the food between
    its fingers on the table -- never lifting it -- purely to farm this term (see task-13-grasp-fix.md): that
    non-grasp used to satisfy ``grasped_mask`` alone and paid the full weight-10 reward with no hold required.
    """
    target = env.scene[bowl_cfg.name].data.root_pos_w.torch.clone()
    target[:, 2] += hover_height
    distance = torch.linalg.vector_norm(env.scene[food_cfg.name].data.root_pos_w.torch - target, dim=-1)
    food_z = env.scene[food_cfg.name].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    grasped = grasped_mask(env, robot_cfg, food_cfg, open_pos=open_pos, closed_pos=closed_pos)
    held = grasped & (food_z > lift_height)
    return held.float() * (1.0 - torch.tanh(distance / std))


def released_in_bowl(
    env: ManagerBasedRLEnv,
    inner_radius: float,
    base_thickness: float,
    rim_height: float,
    item_radius: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """1.0 while the food is inside the bowl and released (same test as the success termination, minus its
    speed and settle requirements). Requiring the release stops a policy from lowering the held food into the
    bowl and hovering there."""
    return released_in_bowl_mask(
        env, inner_radius, base_thickness, rim_height, item_radius, robot_cfg, open_pos, closed_pos, food_cfg, bowl_cfg
    ).float()


def return_home(
    env: ManagerBasedRLEnv,
    std: float,
    inner_radius: float,
    base_thickness: float,
    rim_height: float,
    item_radius: float,
    robot_cfg: SceneEntityCfg,
    open_pos: float,
    closed_pos: float,
    home_pos: tuple[float, float, float],
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """While the food is released inside the bowl: ``1 - tanh(d / std)``, ``d`` = the TCP's distance [m] to its home
    position (``tcp_home_distance``, the same distance the ``success_requires_home`` check uses);
    0 otherwise. Teaches finishing the job: after placing, go back home."""
    placed = released_in_bowl_mask(
        env, inner_radius, base_thickness, rim_height, item_radius, robot_cfg, open_pos, closed_pos, food_cfg, bowl_cfg
    )
    return placed.float() * (1.0 - torch.tanh(tcp_home_distance(env, home_pos, ee_frame_cfg) / std))


def termination_indicator(env: ManagerBasedRLEnv, term_names: list[str]) -> torch.Tensor:
    """1.0 where any of the named termination terms fired this step."""
    fired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in term_names:
        fired |= env.termination_manager.get_term(name)
    return fired.float()


def bowl_disturbance(
    env: ManagerBasedRLEnv,
    belt_event_name: str = "reset_belt",
    pallet_cfg: SceneEntityCfg = SceneEntityCfg("pallet"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """Distance [m] the bowl has been pushed away from where the pallet carries it."""
    belt = env.event_manager.get_term_cfg(belt_event_name).func
    pallet = env.scene[pallet_cfg.name].data
    plate_x = pallet.root_pos_w.torch[:, 0] + pallet.joint_pos.torch[:, belt.joint_ids[0]]
    expected = torch.stack([plate_x + belt.offset[:, 0], pallet.root_pos_w.torch[:, 1] + belt.offset[:, 1]], dim=-1)
    bowl_xy = env.scene[bowl_cfg.name].data.root_pos_w.torch[:, :2]
    return torch.linalg.vector_norm(bowl_xy - expected, dim=-1)
