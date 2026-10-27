"""Deployable geometry-only primitives for dynamic grasping.

The classes in this module intentionally depend only on pose candidates,
robot state, and gripper encoders.  Contact, force, object velocity, tactile,
current, and torque signals are never controller inputs.  Simulator contact
may still be used outside this module as a training label or acceptance
metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StableTargetConfig:
    """Filtering and continuity gates for a stream of selected 6D grasps."""

    translation_alpha: float = 0.45
    rotation_alpha: float = 0.35
    max_translation_jump: float = 0.02
    max_rotation_jump: float = 0.35
    stable_translation: float = 0.004
    stable_rotation: float = 0.08
    stable_steps: int = 4
    rotation_weight: float = 0.05
    switch_margin: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.translation_alpha <= 1.0 or not 0.0 < self.rotation_alpha <= 1.0:
            raise ValueError("target filter alphas must be in (0, 1]")
        if min(
            self.max_translation_jump,
            self.max_rotation_jump,
            self.stable_translation,
            self.stable_rotation,
            self.rotation_weight,
            self.switch_margin,
        ) <= 0.0:
            raise ValueError("target continuity thresholds must be positive")
        if self.stable_steps < 1:
            raise ValueError("stable_steps must be >= 1")


class StableGraspTarget:
    """Track a temporally consistent grasp target without estimating speed.

    ``pose_delta`` is the current candidate error in the same cell/world frame
    used by the end-effector action.  Translation is converted to an absolute
    target before filtering, so robot motion is not mistaken for target
    motion.  Rotation deltas are filtered locally; grasp-region discontinuity
    resets the readiness counter instead of being silently accepted.
    """

    def __init__(self, config: StableTargetConfig = StableTargetConfig()):
        self.config = config
        self.position: torch.Tensor | None = None
        self.rotation_delta: torch.Tensor | None = None
        self.initialized: torch.Tensor | None = None
        self.stable_count: torch.Tensor | None = None
        self.selected_index: torch.Tensor | None = None

    def _ensure(self, pose_delta: torch.Tensor, ee_pos: torch.Tensor) -> None:
        if pose_delta.ndim not in (2, 3) or pose_delta.shape[-1] != 6:
            raise ValueError("pose_delta must have shape (N, 6) or (N, C, 6)")
        if ee_pos.shape != (pose_delta.shape[0], 3):
            raise ValueError("ee_pos must have shape (N, 3)")
        n = pose_delta.shape[0]
        if self.position is None or self.position.shape[0] != n or self.position.device != pose_delta.device:
            self.position = torch.zeros(n, 3, dtype=pose_delta.dtype, device=pose_delta.device)
            self.rotation_delta = torch.zeros(n, 3, dtype=pose_delta.dtype, device=pose_delta.device)
            self.initialized = torch.zeros(n, dtype=torch.bool, device=pose_delta.device)
            self.stable_count = torch.zeros(n, dtype=torch.long, device=pose_delta.device)
            self.selected_index = torch.zeros(n, dtype=torch.long, device=pose_delta.device)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if self.position is None:
            return
        if mask is None:
            self.position.zero_()
            self.rotation_delta.zero_()
            self.initialized.zero_()
            self.stable_count.zero_()
            self.selected_index.zero_()
            return
        rows = mask.reshape(mask.shape[0], -1).any(-1)
        self.position[rows] = 0.0
        self.rotation_delta[rows] = 0.0
        self.initialized[rows] = False
        self.stable_count[rows] = 0
        self.selected_index[rows] = 0

    def update(
        self, pose_delta: torch.Tensor, ee_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure(pose_delta, ee_pos)
        if pose_delta.ndim == 3:
            score = (
                torch.linalg.vector_norm(pose_delta[..., :3], dim=-1)
                + self.config.rotation_weight
                * torch.linalg.vector_norm(pose_delta[..., 3:], dim=-1)
            )
            best = score.argmin(dim=1)
            rows = torch.arange(pose_delta.shape[0], device=pose_delta.device)
            current_score = score[rows, self.selected_index]
            best_score = score[rows, best]
            switch = (~self.initialized) | (best_score + self.config.switch_margin < current_score)
            self.selected_index = torch.where(switch, best, self.selected_index)
            pose_delta = pose_delta[rows, self.selected_index]
        raw_position = ee_pos + pose_delta[:, :3]
        raw_rotation = pose_delta[:, 3:]
        fresh = ~self.initialized
        self.position[fresh] = raw_position[fresh]
        self.rotation_delta[fresh] = raw_rotation[fresh]
        self.initialized[fresh] = True

        translation_innovation = torch.linalg.vector_norm(raw_position - self.position, dim=-1)
        rotation_innovation = torch.linalg.vector_norm(raw_rotation - self.rotation_delta, dim=-1)
        continuous = (
            (translation_innovation <= self.config.max_translation_jump)
            & (rotation_innovation <= self.config.max_rotation_jump)
        )
        stable_sample = (
            (translation_innovation <= self.config.stable_translation)
            & (rotation_innovation <= self.config.stable_rotation)
        )
        self.stable_count = torch.where(
            stable_sample,
            self.stable_count + 1,
            torch.zeros_like(self.stable_count),
        )

        update = continuous & ~fresh
        ta = self.config.translation_alpha
        ra = self.config.rotation_alpha
        self.position[update] = (1.0 - ta) * self.position[update] + ta * raw_position[update]
        self.rotation_delta[update] = (
            (1.0 - ra) * self.rotation_delta[update] + ra * raw_rotation[update]
        )

        # A large jump starts a new region immediately but cannot be declared
        # stable in the same control step.
        jumped = ~continuous & ~fresh
        self.position[jumped] = raw_position[jumped]
        self.rotation_delta[jumped] = raw_rotation[jumped]
        self.stable_count[jumped] = 0

        filtered_delta = torch.cat((self.position - ee_pos, self.rotation_delta), dim=-1)
        ready = self.stable_count >= self.config.stable_steps
        return filtered_delta, ready, continuous


@dataclass(frozen=True)
class SphereGraspSkillConfig:
    """Parameters for hover, align, close, hold, then lift."""

    action_scale: float = 0.02
    intercept_x: float | None = None
    intercept_lead_distance: float = 0.05
    intercept_align_lead_distance: float = 0.04
    hover_height: float = 0.08
    pre_descent_height: float | None = None
    hover_translation_threshold: float = 0.012
    grasp_translation_threshold: float = 0.007
    rotation_threshold: float = 0.15
    alignment_steps: int = 4
    close_steps: int = 16
    hold_steps: int = 10
    lift_steps: int = 30
    lift_timeout_steps: int = 150
    lift_height: float = 0.04
    lift_tolerance: float = 0.003
    lift_target_slack: float | None = None
    lift_detach_height: float | None = None
    lift_xy_gain: float | None = None
    max_lift_xy_step: float | None = None
    preserve_lift_relative_xy: bool = False
    target_xy_feedforward_gain: float = 0.0
    max_target_xy_feedforward_step: float | None = None
    align_target_xy_feedforward: bool = False
    close_requires_raw_alignment: bool = False
    max_close_hold_xy_step: float | None = None
    latch_lift_xy_feedforward: bool = False
    lift_feedforward_x_only: bool = False
    lift_feedforward_decay_steps: int | None = None
    lift_geometry_relatch_height: float | None = None
    lift_relative_z_gain: float | None = None
    settled_coupling_reference: bool = False
    position_gain: float = 1.5
    align_position_gain: float = 4.0
    rotation_gain: float = 0.5
    max_hover_position_step: float = 0.02
    max_hover_rotation_step: float = 0.02
    align_fine_distance: float = 0.02
    max_align_position_step: float = 0.02
    max_align_fine_position_step: float = 0.008
    max_position_step: float = 0.004
    max_rotation_step: float = 0.01
    # Franka's continuous gripper mapping reaches the validated blocked
    # aperture near -0.15.  Full-scale -1.0 over-closes, creates 20+ N pad
    # forces, and can squeeze a sphere out even though both contacts fire.
    close_action: float = -0.15

    def __post_init__(self) -> None:
        if min(
            self.action_scale,
            self.intercept_lead_distance,
            self.hover_height,
            self.hover_translation_threshold,
            self.grasp_translation_threshold,
            self.rotation_threshold,
            self.lift_height,
            self.lift_tolerance,
            self.position_gain,
            self.align_position_gain,
            self.rotation_gain,
            self.max_hover_position_step,
            self.max_hover_rotation_step,
            self.align_fine_distance,
            self.max_align_position_step,
            self.max_align_fine_position_step,
            self.max_position_step,
            self.max_rotation_step,
        ) <= 0.0:
            raise ValueError("skill distances, gains, and scales must be positive")
        if self.pre_descent_height is not None:
            if not 0.0 < self.intercept_align_lead_distance < self.intercept_lead_distance:
                raise ValueError(
                    "intercept_align_lead_distance must be positive and below intercept_lead_distance"
                )
            if not 0.0 < self.pre_descent_height < self.hover_height:
                raise ValueError("pre_descent_height must be positive and below hover_height")
        if self.lift_target_slack is not None and self.lift_target_slack <= 0.0:
            raise ValueError("lift_target_slack must be null or positive")
        if self.lift_detach_height is not None:
            if not 0.0 < self.lift_detach_height < self.lift_height:
                raise ValueError("lift_detach_height must be null or between zero and lift_height")
        if self.lift_xy_gain is not None and self.lift_xy_gain <= 0.0:
            raise ValueError("lift_xy_gain must be null or positive")
        if self.max_lift_xy_step is not None and self.max_lift_xy_step <= 0.0:
            raise ValueError("max_lift_xy_step must be null or positive")
        if self.target_xy_feedforward_gain < 0.0:
            raise ValueError("target_xy_feedforward_gain must be nonnegative")
        if (
            self.max_target_xy_feedforward_step is not None
            and self.max_target_xy_feedforward_step <= 0.0
        ):
            raise ValueError("max_target_xy_feedforward_step must be null or positive")
        if self.max_close_hold_xy_step is not None and self.max_close_hold_xy_step <= 0.0:
            raise ValueError("max_close_hold_xy_step must be null or positive")
        if self.lift_feedforward_decay_steps is not None and self.lift_feedforward_decay_steps < 1:
            raise ValueError("lift_feedforward_decay_steps must be null or >= 1")
        if self.lift_geometry_relatch_height is not None and not (
            0.0 < self.lift_geometry_relatch_height < self.lift_height
        ):
            raise ValueError(
                "lift_geometry_relatch_height must be null or between zero and lift_height"
            )
        if self.lift_relative_z_gain is not None and self.lift_relative_z_gain <= 0.0:
            raise ValueError("lift_relative_z_gain must be null or positive")
        if min(
            self.alignment_steps,
            self.close_steps,
            self.hold_steps,
            self.lift_steps,
            self.lift_timeout_steps,
        ) < 1:
            raise ValueError("skill phase lengths must be >= 1")
        if self.lift_timeout_steps < self.lift_steps:
            raise ValueError("lift_timeout_steps must be >= lift_steps")
        if not -1.0 <= self.close_action < 1.0:
            raise ValueError("close_action must be in [-1, 1)")


class StableSphereGraspSkill:
    """Speed-free dynamic-sphere grasp skill with explicit phase ownership.

    Phases are ``hover -> align -> close -> hold -> lift -> done``.  The
    controller follows the current filtered grasp target throughout close and
    hold, then keeps XY visual servoing while lifting.  Finger aperture is an
    actuator input only; it is never interpreted as grasp success.
    """

    HOVER, ALIGN, CLOSE, HOLD, LIFT, DONE = range(6)
    PHASE_NAMES = ("hover", "align", "close", "hold", "lift", "done")

    def __init__(
        self,
        config: SphereGraspSkillConfig = SphereGraspSkillConfig(),
        target_config: StableTargetConfig = StableTargetConfig(),
    ):
        self.config = config
        self.target = StableGraspTarget(target_config)
        self.phase: torch.Tensor | None = None
        self.phase_step: torch.Tensor | None = None
        self.alignment_count: torch.Tensor | None = None
        self.lift_anchor_z: torch.Tensor | None = None
        self.lift_relative_xy: torch.Tensor | None = None
        self.lift_relative_z: torch.Tensor | None = None
        self.lift_geometry_relatched: torch.Tensor | None = None
        self.lift_xy_feedforward: torch.Tensor | None = None

    def _ensure(self, action: torch.Tensor) -> None:
        if action.ndim != 2 or action.shape[-1] != 7:
            raise ValueError("skill action must have shape (N, 7)")
        n = action.shape[0]
        if self.phase is None or self.phase.shape[0] != n or self.phase.device != action.device:
            self.phase = torch.full((n,), self.HOVER, dtype=torch.long, device=action.device)
            self.phase_step = torch.zeros(n, dtype=torch.long, device=action.device)
            self.alignment_count = torch.zeros(n, dtype=torch.long, device=action.device)
            self.lift_anchor_z = torch.zeros(n, dtype=action.dtype, device=action.device)
            self.lift_relative_xy = torch.zeros(n, 2, dtype=action.dtype, device=action.device)
            self.lift_relative_z = torch.zeros(n, dtype=action.dtype, device=action.device)
            self.lift_geometry_relatched = torch.zeros(n, dtype=torch.bool, device=action.device)
            self.lift_xy_feedforward = torch.zeros(n, 2, dtype=action.dtype, device=action.device)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        self.target.reset(mask)
        if self.phase is None:
            return
        optional_state = (
            self.phase_step,
            self.alignment_count,
            self.lift_anchor_z,
            self.lift_relative_xy,
            self.lift_relative_z,
            self.lift_geometry_relatched,
            self.lift_xy_feedforward,
        )
        if mask is None:
            self.phase.fill_(self.HOVER)
            for state in optional_state:
                if state is not None:
                    state.zero_()
            return
        rows = mask.reshape(mask.shape[0], -1).any(-1)
        self.phase[rows] = self.HOVER
        for state in optional_state:
            if state is not None:
                state[rows] = 0

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        return x.square() * (3.0 - 2.0 * x)

    def __call__(
        self,
        action: torch.Tensor,
        pose_delta: torch.Tensor,
        ee_pos: torch.Tensor,
        finger_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure(action)
        if finger_pos.shape[0] != action.shape[0]:
            raise ValueError("finger_pos batch must match action batch")
        previous_target_position = (
            None if self.target.position is None else self.target.position.clone()
        )
        previous_target_valid = (
            None if self.target.initialized is None else self.target.initialized.clone()
        )
        filtered, target_ready, target_continuous = self.target.update(pose_delta, ee_pos)
        if pose_delta.ndim == 3:
            rows = torch.arange(pose_delta.shape[0], device=pose_delta.device)
            raw_selected = pose_delta[rows, self.target.selected_index]
        else:
            raw_selected = pose_delta
        cfg = self.config
        out = action.clone()
        out.zero_()
        out[:, -1] = 1.0

        # A pure position servo commands zero XY motion once the desired
        # sphere-to-TCP offset is reached, even though the sphere is still
        # translating on the conveyor.  Feed forward the *observed target
        # displacement* between consecutive frames so CLOSE/HOLD/LIFT can
        # keep moving with it.  This is geometry history, not a belt/object
        # velocity input; discontinuous target jumps are explicitly ignored.
        target_xy_feedforward = torch.zeros_like(ee_pos[:, :2])
        if (
            cfg.target_xy_feedforward_gain > 0.0
            and previous_target_position is not None
            and previous_target_valid is not None
        ):
            valid_motion = previous_target_valid & target_continuous
            target_xy_feedforward[valid_motion] = (
                cfg.target_xy_feedforward_gain
                * (self.target.position[valid_motion, :2] - previous_target_position[valid_motion, :2])
            )
            feedforward_limit = (
                cfg.max_position_step
                if cfg.max_target_xy_feedforward_step is None
                else cfg.max_target_xy_feedforward_step
            )
            target_xy_feedforward.clamp_(-feedforward_limit, feedforward_limit)

        rotation_error = torch.linalg.vector_norm(filtered[:, 3:], dim=-1)
        hover_delta = filtered[:, :3].clone()
        hover_delta[:, 2] += cfg.hover_height
        intercept_ready = torch.ones_like(target_ready)
        if cfg.intercept_x is not None:
            # Wait above a fixed conveyor interception line instead of
            # chasing the object from its spawn point.  The trigger uses only
            # the current vision-derived grasp position, never speed.
            if cfg.pre_descent_height is None:
                hover_delta[:, 0] = float(cfg.intercept_x) - ee_pos[:, 0]
                intercept_ready = self.target.position[:, 0] >= (
                    float(cfg.intercept_x) - cfg.intercept_lead_distance
                )
            else:
                from pickplace.intercept import progressive_intercept_hover_delta

                hover_delta, _ = progressive_intercept_hover_delta(
                    ee_pos,
                    filtered[:, :3],
                    cfg.intercept_x,
                    cfg.intercept_lead_distance,
                    cfg.intercept_align_lead_distance,
                    cfg.hover_height,
                    cfg.pre_descent_height,
                )
                intercept_ready = self.target.position[:, 0] >= (
                    float(cfg.intercept_x) - cfg.intercept_align_lead_distance
                )

        hover = self.phase.eq(self.HOVER)
        # In the progressive high-speed schedule, HOVER is a coarse spatial
        # staging phase: descent and wrist alignment run concurrently.  Do
        # not hold the hand at the interception line until the wrist has
        # completely converged, because that turns orientation settling into
        # several seconds of conveyor latency.  ALIGN continues commanding
        # rotation, and CLOSE still requires the original strict rotation
        # threshold together with the 7 mm translation threshold.
        hover_rotation_ready = rotation_error <= cfg.rotation_threshold
        if cfg.pre_descent_height is not None:
            hover_rotation_ready = torch.ones_like(hover_rotation_ready)
        hover_aligned = (
            target_ready
            & intercept_ready
            & (torch.linalg.vector_norm(hover_delta, dim=-1) <= cfg.hover_translation_threshold)
            & hover_rotation_ready
        )
        self.alignment_count = torch.where(
            hover & hover_aligned,
            self.alignment_count + 1,
            torch.where(hover, torch.zeros_like(self.alignment_count), self.alignment_count),
        )
        to_align = hover & (self.alignment_count >= cfg.alignment_steps)
        self.phase[to_align] = self.ALIGN
        self.phase_step[to_align] = 0
        self.alignment_count[to_align] = 0

        align = self.phase.eq(self.ALIGN)
        grasp_aligned = (
            target_ready
            & (torch.linalg.vector_norm(filtered[:, :3], dim=-1) <= cfg.grasp_translation_threshold)
            & (rotation_error <= cfg.rotation_threshold)
        )
        if cfg.close_requires_raw_alignment:
            # Filtering is useful for target continuity but introduces a
            # downstream lag on a fast conveyor.  A close transition must not
            # be authorized solely by that stale target: require the current
            # selected pose error to satisfy the same strict gate as well.
            grasp_aligned &= (
                torch.linalg.vector_norm(raw_selected[:, :3], dim=-1)
                <= cfg.grasp_translation_threshold
            ) & (
                torch.linalg.vector_norm(raw_selected[:, 3:], dim=-1)
                <= cfg.rotation_threshold
            )
        self.alignment_count = torch.where(
            align & grasp_aligned,
            self.alignment_count + 1,
            torch.where(align, torch.zeros_like(self.alignment_count), self.alignment_count),
        )
        to_close = align & (self.alignment_count >= cfg.alignment_steps)
        self.phase[to_close] = self.CLOSE
        self.phase_step[to_close] = 0
        self.alignment_count[to_close] = 0

        phase = self.phase.clone()
        commanded_delta = filtered[:, :3].clone()
        commanded_delta[phase == self.HOVER] = hover_delta[phase == self.HOVER]
        position_step = (cfg.position_gain * commanded_delta).clamp(
            -cfg.max_position_step, cfg.max_position_step
        )
        post_close_tracking = (phase == self.CLOSE) | (phase == self.HOLD)
        close_hold_xy_limit = (
            cfg.max_position_step
            if cfg.max_close_hold_xy_step is None
            else cfg.max_close_hold_xy_step
        )
        position_step[post_close_tracking, :2] = (
            cfg.position_gain * commanded_delta[post_close_tracking, :2]
            + target_xy_feedforward[post_close_tracking]
        ).clamp(-close_hold_xy_limit, close_hold_xy_limit)
        rotation_step = (cfg.rotation_gain * filtered[:, 3:]).clamp(
            -cfg.max_rotation_step, cfg.max_rotation_step
        )
        tracking = (phase == self.HOVER) | (phase == self.ALIGN) | (phase == self.CLOSE) | (phase == self.HOLD)
        out[tracking, :3] = (position_step[tracking] / cfg.action_scale).clamp(-1.0, 1.0)
        out[tracking, 3:6] = (rotation_step[tracking] / cfg.action_scale).clamp(-1.0, 1.0)
        hover_position_step = (cfg.position_gain * hover_delta).clamp(
            -cfg.max_hover_position_step, cfg.max_hover_position_step
        )
        hover_rotation_step = (cfg.rotation_gain * filtered[:, 3:]).clamp(
            -cfg.max_hover_rotation_step, cfg.max_hover_rotation_step
        )
        out[phase == self.HOVER, :3] = (
            hover_position_step[phase == self.HOVER] / cfg.action_scale
        ).clamp(-1.0, 1.0)
        out[phase == self.HOVER, 3:6] = (
            hover_rotation_step[phase == self.HOVER] / cfg.action_scale
        ).clamp(-1.0, 1.0)

        # Use centimetre-scale motion for the long descent from the hover
        # point, then retain millimetre-scale servoing near the sphere.
        # CLOSE/HOLD always use the fine limit above.
        align = phase == self.ALIGN
        align_far = align & (
            torch.linalg.vector_norm(commanded_delta, dim=-1) > cfg.align_fine_distance
        )
        align_near = align & ~align_far
        align_position_step = (cfg.align_position_gain * commanded_delta).clamp(
            -cfg.max_align_position_step, cfg.max_align_position_step
        )
        align_fine_step = (cfg.align_position_gain * commanded_delta).clamp(
            -cfg.max_align_fine_position_step, cfg.max_align_fine_position_step
        )
        if cfg.align_target_xy_feedforward:
            # Match the target's observed frame-to-frame displacement before
            # closing.  This uses pose history only; no velocity channel is
            # exposed to the actor or controller.
            align_position_step[:, :2] = (
                cfg.align_position_gain * commanded_delta[:, :2]
                + target_xy_feedforward
            ).clamp(-cfg.max_align_position_step, cfg.max_align_position_step)
            align_fine_step[:, :2] = (
                cfg.align_position_gain * commanded_delta[:, :2]
                + target_xy_feedforward
            ).clamp(-cfg.max_align_fine_position_step, cfg.max_align_fine_position_step)
        out[align_far, :3] = (
            align_position_step[align_far] / cfg.action_scale
        ).clamp(-1.0, 1.0)
        out[align_near, :3] = (
            align_fine_step[align_near] / cfg.action_scale
        ).clamp(-1.0, 1.0)

        closed = (phase == self.CLOSE) | (phase == self.HOLD) | (phase == self.LIFT) | (phase == self.DONE)
        out[closed, -1] = cfg.close_action

        close = phase == self.CLOSE
        self.phase_step[close] += 1
        to_hold = close & (self.phase_step >= cfg.close_steps)
        self.phase[to_hold] = self.HOLD
        self.phase_step[to_hold] = 0

        hold = self.phase == self.HOLD
        self.phase_step[hold] += 1
        to_lift = hold & (self.phase_step >= cfg.hold_steps)
        self.lift_anchor_z[to_lift] = ee_pos[to_lift, 2]
        # Preserve the stable, non-zero sphere-to-TCP offset captured after
        # closure.  Driving this offset to zero would push the already-held
        # sphere laterally through the fingers.  Only pose geometry is used.
        self.lift_relative_xy[to_lift] = filtered[to_lift, :2]
        self.lift_relative_z[to_lift] = filtered[to_lift, 2]
        self.lift_geometry_relatched[to_lift] = False
        self.lift_xy_feedforward[to_lift] = target_xy_feedforward[to_lift]
        if cfg.lift_feedforward_x_only:
            # A straight conveyor has a known transport axis.  Contact
            # transients can create a one-frame lateral target displacement;
            # never latch that disturbance as a persistent lift command.
            self.lift_xy_feedforward[to_lift, 1] = 0.0
        self.phase[to_lift] = self.LIFT
        self.phase_step[to_lift] = 0

        lift = self.phase == self.LIFT
        if lift.any():
            if cfg.lift_geometry_relatch_height is not None:
                # The belt supports the sphere during HOLD.  After a small
                # peel, gravity loads it against the fingers and the stable
                # relative pose changes.  Latch that loaded geometry once so
                # the controller does not keep chasing the unloaded pose and
                # roll the sphere through the pads.  The trigger uses only
                # TCP displacement and the current grasp geometry.
                relatch = (
                    lift
                    & ~self.lift_geometry_relatched
                    & (
                        ee_pos[:, 2]
                        >= self.lift_anchor_z + float(cfg.lift_geometry_relatch_height)
                    )
                )
                self.lift_relative_xy[relatch] = filtered[relatch, :2]
                self.lift_relative_z[relatch] = filtered[relatch, 2]
                self.lift_geometry_relatched[relatch] = True
            # Keep following the grasp target laterally, but use a monotonic
            # TCP height command so a slipping sphere cannot pull the hand
            # back down and conceal failure.
            lift_xy_gain = cfg.position_gain if cfg.lift_xy_gain is None else cfg.lift_xy_gain
            lift_xy_limit = (
                cfg.max_position_step
                if cfg.max_lift_xy_step is None
                else cfg.max_lift_xy_step
            )
            lift_xy_error = filtered[:, :2]
            if cfg.preserve_lift_relative_xy:
                lift_xy_error = lift_xy_error - self.lift_relative_xy
            xy_step = (lift_xy_gain * lift_xy_error).clamp(
                -lift_xy_limit, lift_xy_limit
            )
            lift_feedforward = (
                self.lift_xy_feedforward
                if cfg.latch_lift_xy_feedforward
                else target_xy_feedforward
            )
            if cfg.latch_lift_xy_feedforward and cfg.lift_feedforward_decay_steps is not None:
                # Match the moving object briefly after closure, then remove
                # the feed-forward smoothly so lift does not integrate one
                # observed displacement forever and drive the arm sideways.
                decay = (
                    1.0
                    - self.phase_step.float() / float(cfg.lift_feedforward_decay_steps)
                ).clamp(0.0, 1.0)
                lift_feedforward = lift_feedforward * decay[:, None]
            xy_step = (xy_step + lift_feedforward).clamp(
                -lift_xy_limit, lift_xy_limit
            )
            progress = ((self.phase_step.float() + 1.0) / cfg.lift_steps).clamp(0.0, 1.0)
            target_z = self.lift_anchor_z + cfg.lift_height * self._smoothstep(progress)
            if cfg.lift_target_slack is not None:
                # Geometry-only lift servo: do not let the TCP outrun the
                # observed grasp target.  If the object lags, vertical motion
                # pauses until it follows; no contact, force, speed, tactile,
                # current, or torque signal enters this controller.
                observed_ceiling = self.target.position[:, 2] + float(cfg.lift_target_slack)
                if cfg.lift_detach_height is None:
                    target_z = torch.minimum(target_z, observed_ceiling)
                else:
                    # First peel the sphere clear of the moving belt.  A
                    # small follower lead from the first millimetre pins the
                    # sphere against the belt instead of lifting it.  After
                    # the TCP has reached the detach height, switch to the
                    # geometry-only follower ceiling.
                    detached = ee_pos[:, 2] >= (
                        self.lift_anchor_z + float(cfg.lift_detach_height)
                    )
                    target_z = torch.where(
                        detached,
                        torch.minimum(target_z, observed_ceiling),
                        target_z,
                    )
            z_step = (cfg.position_gain * (target_z - ee_pos[:, 2])).clamp(
                -cfg.max_position_step, cfg.max_position_step
            )
            if cfg.lift_relative_z_gain is not None:
                # Keep the nominal lift trajectory as feed-forward, then
                # gently reduce upward motion when the observed grasp target
                # falls behind its captured post-close Z geometry.  Unlike a
                # hard target ceiling this can still peel the sphere off the
                # moving belt and continue upward.
                relative_z_error = filtered[:, 2] - self.lift_relative_z
                z_step = (
                    z_step + float(cfg.lift_relative_z_gain) * relative_z_error
                ).clamp(-cfg.max_position_step, cfg.max_position_step)
            out[lift, :2] = (xy_step[lift] / cfg.action_scale).clamp(-1.0, 1.0)
            out[lift, 2] = (z_step[lift] / cfg.action_scale).clamp(-1.0, 1.0)
            out[lift, 3:6] = 0.0
            self.phase_step[lift] += 1

        final_lift_error = (self.lift_anchor_z + cfg.lift_height - ee_pos[:, 2]).abs()
        reached_lift_pose = (self.phase_step >= cfg.lift_steps) & (final_lift_error <= cfg.lift_tolerance)
        lift_timed_out = self.phase_step >= cfg.lift_timeout_steps
        to_done = lift & (reached_lift_pose | lift_timed_out)
        self.phase[to_done] = self.DONE
        self.phase_step[to_done] = 0
        done = self.phase == self.DONE
        if done.any():
            out[done, :6] = 0.0

        return out, self.phase.clone(), filtered


class StableSphereResidualSkill:
    """Validated geometric sphere skill plus a tightly bounded PPO residual.

    The geometric controller owns the complete hover/align/close/hold/lift
    sequence.  The learned action can make only small arm corrections during
    hover and alignment; the gripper and the contact-critical phases remain
    entirely under the deterministic skill.  Limits are expressed in the
    normalized action space (and are therefore multiplied by
    ``SphereGraspSkillConfig.action_scale`` by the Isaac action term).
    """

    def __init__(
        self,
        skill_config: SphereGraspSkillConfig = SphereGraspSkillConfig(),
        target_config: StableTargetConfig = StableTargetConfig(),
        translation_residual_limit: float = 0.10,
        rotation_residual_limit: float = 0.05,
    ):
        if not 0.0 <= translation_residual_limit <= 1.0:
            raise ValueError("translation_residual_limit must be in [0, 1]")
        if not 0.0 <= rotation_residual_limit <= 1.0:
            raise ValueError("rotation_residual_limit must be in [0, 1]")
        self.baseline = StableSphereGraspSkill(skill_config, target_config)
        self.translation_residual_limit = float(translation_residual_limit)
        self.rotation_residual_limit = float(rotation_residual_limit)

    @property
    def phase(self) -> torch.Tensor | None:
        return self.baseline.phase

    def reset(self, mask: torch.Tensor | None = None) -> None:
        self.baseline.reset(mask)

    def __call__(
        self,
        residual_action: torch.Tensor,
        pose_delta: torch.Tensor,
        ee_pos: torch.Tensor,
        finger_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nominal, phase, filtered = self.baseline(
            torch.zeros_like(residual_action), pose_delta, ee_pos, finger_pos
        )
        correction = torch.zeros_like(residual_action)
        learnable = phase.le(self.baseline.ALIGN)
        correction[learnable, :3] = (
            residual_action[learnable, :3].clamp(-1.0, 1.0)
            * self.translation_residual_limit
        )
        correction[learnable, 3:6] = (
            residual_action[learnable, 3:6].clamp(-1.0, 1.0)
            * self.rotation_residual_limit
        )
        executed = (nominal + correction).clamp(-1.0, 1.0)
        # The policy never owns finger closure, even while approaching.
        executed[:, -1] = nominal[:, -1]
        return executed, phase, filtered, nominal


@dataclass(frozen=True)
class SphereTransportPlaceConfig:
    """Geometry-only handoff after the validated moving-sphere lift.

    The bowl pose is a calibrated/vision-derived deployable input.  No object
    velocity, contact, force, tactile, current, or torque value is consumed.
    """

    action_scale: float = 0.02
    clear_height: float = 0.20
    bowl_hover_height: float = 0.20
    release_height: float = 0.11
    position_gain: float = 4.0
    # The action term scales unit translation commands by 2 cm.  A 1.6 cm
    # command cap produces roughly 8 cm/s measured TCP motion in this setup.
    # A command slew limiter prevents impulsive waypoint transitions.
    max_position_step: float = 0.016
    max_translation_action_delta: float = 0.04
    max_lower_xy_step: float = 0.002
    position_deadband: float = 0.001
    clear_tolerance: float = 0.012
    hover_tolerance: float = 0.018
    release_tolerance: float = 0.012
    stable_steps: int = 3
    release_steps: int = 28
    close_action: float = -0.15
    open_action: float = 1.0
    # Optional complete-task tail.  ``None`` preserves the frozen G1
    # moving-sphere/static-bowl behavior byte-for-byte.
    home_position: tuple[float, float, float] | None = None
    retreat_height: float = 0.20
    retreat_tolerance: float = 0.015
    home_tolerance: float = 0.025
    max_home_position_step: float = 0.016

    def __post_init__(self) -> None:
        if min(
            self.action_scale,
            self.clear_height,
            self.bowl_hover_height,
            self.release_height,
            self.position_gain,
            self.max_position_step,
            self.max_translation_action_delta,
            self.max_lower_xy_step,
            self.position_deadband,
            self.clear_tolerance,
            self.hover_tolerance,
            self.release_tolerance,
        ) <= 0.0:
            raise ValueError("transport/place distances, gains, and scales must be positive")
        if self.release_height >= self.bowl_hover_height:
            raise ValueError("release_height must be below bowl_hover_height")
        if min(self.stable_steps, self.release_steps) < 1:
            raise ValueError("transport/place step counts must be >= 1")
        if not -1.0 <= self.close_action < self.open_action <= 1.0:
            raise ValueError("transport/place gripper actions must satisfy -1 <= close < open <= 1")
        if self.home_position is not None:
            if len(self.home_position) != 3:
                raise ValueError("home_position must contain three values")
            if min(
                self.retreat_height,
                self.retreat_tolerance,
                self.home_tolerance,
                self.max_home_position_step,
            ) <= 0.0:
                raise ValueError("return-home distances must be positive")


class StableSpherePickPlaceSkill:
    """Validated grasp/lift followed by safe transport and open-loop release.

    The grasp controller remains byte-for-byte the G0 controller.  Once its
    LIFT phase reaches DONE, this wrapper first raises vertically above the
    conveyor's lateral termination height, then moves over the static bowl,
    lowers, and opens.  The explicit vertical-clear phase prevents a valid
    grasp from being misclassified as leaving the belt during lateral motion.
    """

    WAIT_GRASP, CLEAR, BOWL_HOVER, LOWER, RELEASE, DONE, RETREAT, HOME = range(8)
    PHASE_NAMES = (
        "wait_grasp", "clear", "bowl_hover", "lower", "release", "done",
        "retreat", "home",
    )

    def __init__(
        self,
        grasp: StableSphereResidualSkill,
        config: SphereTransportPlaceConfig = SphereTransportPlaceConfig(),
    ):
        self.grasp = grasp
        self.config = config
        self.phase: torch.Tensor | None = None
        self.phase_step: torch.Tensor | None = None
        self.stable_count: torch.Tensor | None = None
        self.clear_xy: torch.Tensor | None = None
        self.translation_action: torch.Tensor | None = None

    def _ensure(self, action: torch.Tensor) -> None:
        if action.ndim != 2 or action.shape[-1] != 7:
            raise ValueError("pick/place skill expects (N, 7) actions")
        n = action.shape[0]
        if self.phase is None or self.phase.shape[0] != n or self.phase.device != action.device:
            self.phase = torch.zeros(n, dtype=torch.long, device=action.device)
            self.phase_step = torch.zeros(n, dtype=torch.long, device=action.device)
            self.stable_count = torch.zeros(n, dtype=torch.long, device=action.device)
            self.clear_xy = torch.zeros(n, 2, dtype=action.dtype, device=action.device)
            self.translation_action = torch.zeros(n, 3, dtype=action.dtype, device=action.device)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        self.grasp.reset(mask)
        if self.phase is None:
            return
        if mask is None:
            rows = torch.ones_like(self.phase, dtype=torch.bool)
        else:
            rows = mask.reshape(mask.shape[0], -1).any(-1)
        self.phase[rows] = self.WAIT_GRASP
        self.phase_step[rows] = 0
        self.stable_count[rows] = 0
        self.clear_xy[rows] = 0.0
        self.translation_action[rows] = 0.0

    def __call__(
        self,
        residual_action: torch.Tensor,
        pose_delta: torch.Tensor,
        ee_pos: torch.Tensor,
        finger_pos: torch.Tensor,
        bowl_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure(residual_action)
        if bowl_pos.shape != ee_pos.shape:
            raise ValueError("bowl_pos must have shape (N, 3)")
        out, grasp_phase, _, _ = self.grasp(
            residual_action, pose_delta, ee_pos, finger_pos
        )
        enter = (self.phase == self.WAIT_GRASP) & (
            grasp_phase == self.grasp.baseline.DONE
        )
        self.clear_xy[enter] = ee_pos[enter, :2]
        self.phase[enter] = self.CLEAR
        self.phase_step[enter] = 0
        self.stable_count[enter] = 0

        cfg = self.config
        phase = self.phase.clone()
        active = (phase >= self.CLEAR) & (phase <= self.RELEASE)
        out[active] = 0.0
        out[active, -1] = cfg.close_action

        targets = ee_pos.clone()
        clear = phase == self.CLEAR
        targets[clear, :2] = self.clear_xy[clear]
        targets[clear, 2] = cfg.clear_height
        hover = phase == self.BOWL_HOVER
        targets[hover] = bowl_pos[hover]
        targets[hover, 2] += cfg.bowl_hover_height
        lower = phase == self.LOWER
        targets[lower] = bowl_pos[lower]
        targets[lower, 2] += cfg.release_height

        moving = clear | hover | lower
        target_error = targets - ee_pos
        step = (cfg.position_gain * target_error).clamp(
            -cfg.max_position_step, cfg.max_position_step
        )
        # Descent is a vertical skill.  Retain only a small XY correction for
        # calibration error instead of allowing a second lateral approach.
        step[lower, :2] = step[lower, :2].clamp(
            -cfg.max_lower_xy_step, cfg.max_lower_xy_step
        )
        step[torch.abs(target_error) <= cfg.position_deadband] = 0.0
        out[moving, :3] = (step[moving] / cfg.action_scale).clamp(-1.0, 1.0)
        # Slew-limit acceleration from rest, but follow braking commands
        # immediately.  Symmetrically slew-limiting deceleration carried the
        # old waypoint velocity through the next waypoint, producing a large
        # overshoot and visible hunting above the bowl.
        desired_translation = out[:, :3].clone()
        translation_delta = (desired_translation - self.translation_action).clamp(
            -cfg.max_translation_action_delta, cfg.max_translation_action_delta
        )
        same_direction = desired_translation * self.translation_action > 0.0
        accelerating = same_direction & (
            torch.abs(desired_translation) > torch.abs(self.translation_action)
        )
        starting = (self.translation_action == 0.0) & (desired_translation != 0.0)
        slew = moving[:, None] & (accelerating | starting)
        next_translation = desired_translation.clone()
        next_translation[slew] = (self.translation_action + translation_delta)[slew]
        self.translation_action[moving] = next_translation[moving]
        out[moving, :3] = self.translation_action[moving]
        self.translation_action[~moving] = 0.0
        # Zero rotational deltas preserve the top-down orientation established
        # by G0 while avoiding an extra orientation handoff transient.
        out[moving, 3:6] = 0.0

        tolerance = torch.full_like(self.stable_count, cfg.release_tolerance, dtype=ee_pos.dtype)
        tolerance[clear] = cfg.clear_tolerance
        tolerance[hover] = cfg.hover_tolerance
        reached = moving & (torch.linalg.vector_norm(targets - ee_pos, dim=-1) <= tolerance)
        self.stable_count = torch.where(
            reached,
            self.stable_count + 1,
            torch.where(moving, torch.zeros_like(self.stable_count), self.stable_count),
        )
        advance = moving & (self.stable_count >= cfg.stable_steps)
        self.phase[advance] += 1
        self.phase_step[advance] = 0
        self.stable_count[advance] = 0
        # Each waypoint starts from rest.  This prevents a lateral BOWL_HOVER
        # command from leaking into the vertical LOWER phase.
        self.translation_action[advance] = 0.0

        release = self.phase == self.RELEASE
        out[release, :6] = 0.0
        out[release, -1] = cfg.open_action
        self.phase_step[release] += 1
        released = release & (self.phase_step >= cfg.release_steps)
        if cfg.home_position is None:
            self.phase[released] = self.DONE
        else:
            self.phase[released] = self.RETREAT
            self.clear_xy[released] = ee_pos[released, :2]
        self.phase_step[released] = 0

        if cfg.home_position is not None:
            retreat = self.phase == self.RETREAT
            home = self.phase == self.HOME
            return_active = retreat | home
            if return_active.any():
                return_targets = ee_pos.clone()
                return_targets[retreat, :2] = self.clear_xy[retreat]
                return_targets[retreat, 2] = cfg.retreat_height
                home_target = ee_pos.new_tensor(cfg.home_position)
                return_targets[home] = home_target
                return_error = return_targets - ee_pos
                return_step = (cfg.position_gain * return_error).clamp(
                    -cfg.max_home_position_step, cfg.max_home_position_step
                )
                return_step[torch.abs(return_error) <= cfg.position_deadband] = 0.0
                out[return_active] = 0.0
                out[return_active, :3] = (
                    return_step[return_active] / cfg.action_scale
                ).clamp(-1.0, 1.0)
                out[return_active, -1] = cfg.open_action
                reached_retreat = retreat & (
                    torch.linalg.vector_norm(return_error, dim=-1) <= cfg.retreat_tolerance
                )
                self.phase[reached_retreat] = self.HOME
                reached_home = home & (
                    torch.linalg.vector_norm(return_error, dim=-1) <= cfg.home_tolerance
                )
                self.phase[reached_home] = self.DONE

        done = self.phase == self.DONE
        out[done] = 0.0
        out[done, -1] = cfg.open_action
        return out, grasp_phase, self.phase.clone()


@dataclass(frozen=True)
class SphereCloseHoldLiftOptionConfig:
    """Geometry-only option entered after the policy reaches a grasp basin."""

    action_scale: float = 0.02
    trigger_translation: float = 0.018
    grasp_translation: float = 0.007
    rotation_threshold: float = 0.15
    alignment_steps: int = 4
    close_steps: int = 16
    hold_steps: int = 10
    lift_steps: int = 30
    lift_timeout_steps: int = 150
    lift_height: float = 0.04
    lift_tolerance: float = 0.003
    align_position_gain: float = 4.0
    position_gain: float = 1.5
    rotation_gain: float = 0.5
    max_align_position_step: float = 0.008
    max_position_step: float = 0.004
    max_rotation_step: float = 0.01
    close_action: float = -0.15

    def __post_init__(self) -> None:
        positive = (
            self.action_scale, self.trigger_translation, self.grasp_translation,
            self.rotation_threshold, self.lift_height, self.lift_tolerance,
            self.align_position_gain, self.position_gain, self.rotation_gain,
            self.max_align_position_step, self.max_position_step, self.max_rotation_step,
        )
        if min(positive) <= 0.0:
            raise ValueError("sphere option distances, gains, and scales must be positive")
        if min(self.alignment_steps, self.close_steps, self.hold_steps, self.lift_steps, self.lift_timeout_steps) < 1:
            raise ValueError("sphere option phase lengths must be >= 1")
        if self.lift_timeout_steps < self.lift_steps:
            raise ValueError("lift_timeout_steps must be >= lift_steps")
        if not -1.0 <= self.close_action < 1.0:
            raise ValueError("close_action must be in [-1, 1)")


class StableSphereCloseHoldLiftOption:
    """Policy-triggered ALIGN->CLOSE->HOLD->LIFT option.

    Before triggering, only the gripper is forced open and the policy owns all
    arm axes. Once a stable 6D candidate enters the trigger basin, this option
    owns the controller until reset. It never reads speed, contact, or force.
    """

    IDLE, ALIGN, CLOSE, HOLD, LIFT, DONE = range(6)
    PHASE_NAMES = ("idle", "align", "close", "hold", "lift", "done")

    def __init__(
        self,
        config: SphereCloseHoldLiftOptionConfig = SphereCloseHoldLiftOptionConfig(),
        target_config: StableTargetConfig = StableTargetConfig(),
    ):
        self.config = config
        self.target = StableGraspTarget(target_config)
        self.phase: torch.Tensor | None = None
        self.phase_step: torch.Tensor | None = None
        self.alignment_count: torch.Tensor | None = None
        self.lift_anchor_z: torch.Tensor | None = None

    def _ensure(self, action: torch.Tensor) -> None:
        if action.ndim != 2 or action.shape[-1] != 7:
            raise ValueError("sphere option action must have shape (N, 7)")
        n = action.shape[0]
        if self.phase is None or self.phase.shape[0] != n or self.phase.device != action.device:
            self.phase = torch.full((n,), self.IDLE, dtype=torch.long, device=action.device)
            self.phase_step = torch.zeros(n, dtype=torch.long, device=action.device)
            self.alignment_count = torch.zeros(n, dtype=torch.long, device=action.device)
            self.lift_anchor_z = torch.zeros(n, dtype=action.dtype, device=action.device)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        self.target.reset(mask)
        if self.phase is None:
            return
        rows = torch.ones_like(self.phase, dtype=torch.bool) if mask is None else mask.reshape(mask.shape[0], -1).any(-1)
        self.phase[rows] = self.IDLE
        self.phase_step[rows] = 0
        self.alignment_count[rows] = 0
        self.lift_anchor_z[rows] = 0.0

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        return x.square() * (3.0 - 2.0 * x)

    def __call__(
        self,
        action: torch.Tensor,
        candidate_pose_deltas: torch.Tensor,
        ee_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure(action)
        filtered, target_ready, _ = self.target.update(candidate_pose_deltas, ee_pos)
        cfg = self.config
        out = action.clone()
        # Policy owns interception, but cannot manufacture an aperture-only
        # grasp before the validated option triggers.
        out[self.phase == self.IDLE, -1] = 1.0

        translation_error = torch.linalg.vector_norm(filtered[:, :3], dim=-1)
        rotation_error = torch.linalg.vector_norm(filtered[:, 3:], dim=-1)
        idle = self.phase == self.IDLE
        trigger_aligned = target_ready & (translation_error <= cfg.trigger_translation) & (
            rotation_error <= cfg.rotation_threshold
        )
        self.alignment_count = torch.where(
            idle & trigger_aligned,
            self.alignment_count + 1,
            torch.where(idle, torch.zeros_like(self.alignment_count), self.alignment_count),
        )
        trigger = idle & (self.alignment_count >= cfg.alignment_steps)
        self.phase[trigger] = self.ALIGN
        self.phase_step[trigger] = 0
        self.alignment_count[trigger] = 0

        align = self.phase == self.ALIGN
        grasp_aligned = target_ready & (translation_error <= cfg.grasp_translation) & (
            rotation_error <= cfg.rotation_threshold
        )
        self.alignment_count = torch.where(
            align & grasp_aligned,
            self.alignment_count + 1,
            torch.where(align, torch.zeros_like(self.alignment_count), self.alignment_count),
        )
        to_close = align & (self.alignment_count >= cfg.alignment_steps)
        self.phase[to_close] = self.CLOSE
        self.phase_step[to_close] = 0
        self.alignment_count[to_close] = 0

        phase = self.phase.clone()
        active = phase > self.IDLE
        tracking = (phase == self.ALIGN) | (phase == self.CLOSE) | (phase == self.HOLD)
        fine_step = (cfg.position_gain * filtered[:, :3]).clamp(-cfg.max_position_step, cfg.max_position_step)
        align_step = (cfg.align_position_gain * filtered[:, :3]).clamp(
            -cfg.max_align_position_step, cfg.max_align_position_step
        )
        rotation_step = (cfg.rotation_gain * filtered[:, 3:]).clamp(
            -cfg.max_rotation_step, cfg.max_rotation_step
        )
        out[tracking, :3] = (fine_step[tracking] / cfg.action_scale).clamp(-1.0, 1.0)
        out[phase == self.ALIGN, :3] = (align_step[phase == self.ALIGN] / cfg.action_scale).clamp(-1.0, 1.0)
        out[tracking, 3:6] = (rotation_step[tracking] / cfg.action_scale).clamp(-1.0, 1.0)

        closed = (phase == self.CLOSE) | (phase == self.HOLD) | (phase == self.LIFT) | (phase == self.DONE)
        out[closed, -1] = cfg.close_action
        close = phase == self.CLOSE
        self.phase_step[close] += 1
        to_hold = close & (self.phase_step >= cfg.close_steps)
        self.phase[to_hold] = self.HOLD
        self.phase_step[to_hold] = 0
        hold = self.phase == self.HOLD
        self.phase_step[hold] += 1
        to_lift = hold & (self.phase_step >= cfg.hold_steps)
        self.lift_anchor_z[to_lift] = ee_pos[to_lift, 2]
        self.phase[to_lift] = self.LIFT
        self.phase_step[to_lift] = 0

        lift = self.phase == self.LIFT
        if lift.any():
            xy_step = (cfg.position_gain * filtered[:, :2]).clamp(-cfg.max_position_step, cfg.max_position_step)
            progress = ((self.phase_step.float() + 1.0) / cfg.lift_steps).clamp(0.0, 1.0)
            target_z = self.lift_anchor_z + cfg.lift_height * self._smoothstep(progress)
            z_step = (cfg.position_gain * (target_z - ee_pos[:, 2])).clamp(-cfg.max_position_step, cfg.max_position_step)
            out[lift, :2] = (xy_step[lift] / cfg.action_scale).clamp(-1.0, 1.0)
            out[lift, 2] = (z_step[lift] / cfg.action_scale).clamp(-1.0, 1.0)
            out[lift, 3:6] = 0.0
            self.phase_step[lift] += 1

        final_error = (self.lift_anchor_z + cfg.lift_height - ee_pos[:, 2]).abs()
        completed = lift & (self.phase_step >= cfg.lift_steps) & (final_error <= cfg.lift_tolerance)
        timed_out = lift & (self.phase_step >= cfg.lift_timeout_steps)
        self.phase[completed | timed_out] = self.DONE
        done = self.phase == self.DONE
        out[done, :6] = 0.0
        out[done, -1] = cfg.close_action
        return out, self.phase.clone(), active
