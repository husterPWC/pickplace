"""Optional low-level safety/stability gates applied after a policy acts.

The policy still produces its native action.  A gate only changes the command
sent to the simulator, which makes the same mechanism usable during rollout,
training, and eventually on a real controller boundary.
"""

from __future__ import annotations

import torch


class GraspHoldGate:
    """Hold a newly grasped object before allowing positive vertical motion.

    For ``hold_steps`` consecutive commands while the observation says the
    object is grasped, positive ``dz`` is capped and the gripper is forced
    closed.  By default horizontal and rotational commands are untouched;
    optional object-relative XY following and rotation freezing can instead
    isolate the new grasp from an immediate transport command.  Once the
    window completes the gate latches open until the grasp is lost.
    """

    def __init__(
        self,
        hold_steps: int,
        max_upward_dz: float = 0.0,
        force_close: bool = True,
        follow_object_xy: bool = False,
        control_dt: float = 0.02,
        action_scale: float = 0.5,
        follow_gain: float = 0.5,
        max_follow_speed: float = 0.15,
        freeze_rotation: bool = False,
    ):
        if hold_steps < 1:
            raise ValueError(f"hold_steps must be >= 1, got {hold_steps}")
        if max_upward_dz < 0:
            raise ValueError(f"max_upward_dz must be >= 0, got {max_upward_dz}")
        if control_dt <= 0 or action_scale <= 0 or follow_gain < 0 or max_follow_speed <= 0:
            raise ValueError("control_dt/action_scale/max_follow_speed must be > 0 and follow_gain must be >= 0")
        self.hold_steps = int(hold_steps)
        self.max_upward_dz = float(max_upward_dz)
        self.force_close = bool(force_close)
        self.follow_object_xy = bool(follow_object_xy)
        self.control_dt = float(control_dt)
        self.action_scale = float(action_scale)
        self.follow_gain = float(follow_gain)
        self.max_follow_speed = float(max_follow_speed)
        self.freeze_rotation = bool(freeze_rotation)
        self.count: torch.Tensor | None = None
        self.ready: torch.Tensor | None = None
        self.anchor_object_to_ee_xy: torch.Tensor | None = None

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if self.count is None:
            return
        if mask is None:
            self.count.zero_()
            self.ready.zero_()
            if self.anchor_object_to_ee_xy is not None:
                self.anchor_object_to_ee_xy.zero_()
        else:
            rows = mask.reshape(mask.shape[0], -1).any(-1)
            self.count[rows] = 0
            self.ready[rows] = False
            if self.anchor_object_to_ee_xy is not None:
                self.anchor_object_to_ee_xy[rows] = 0.0

    def __call__(
        self,
        action: torch.Tensor,
        grasped: torch.Tensor,
        object_pos: torch.Tensor | None = None,
        ee_pos: torch.Tensor | None = None,
        object_lin_vel: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action.shape[-1] != 7:
            raise ValueError(f"grasp hold gate requires 7-D ee_delta_pose actions, got {tuple(action.shape)}")
        held = grasped.reshape(grasped.shape[0], -1).any(-1).bool()
        if self.count is None or self.count.shape != held.shape or self.count.device != held.device:
            self.count = torch.zeros_like(held, dtype=torch.long)
            self.ready = torch.zeros_like(held, dtype=torch.bool)
            self.anchor_object_to_ee_xy = torch.zeros(held.shape[0], 2, dtype=action.dtype, device=held.device)

        # Losing the object creates a fresh state for a later re-grasp.
        self.count = torch.where(held, self.count, torch.zeros_like(self.count))
        self.ready &= held
        active = held & ~self.ready

        gated = action.clone()
        if active.any():
            gated[active, 2] = gated[active, 2].clamp(max=self.max_upward_dz)
            if self.force_close:
                gated[active, -1] = -1.0
            if self.freeze_rotation:
                gated[active, 3:6] = 0.0
            if self.follow_object_xy:
                if object_pos is None or ee_pos is None or object_lin_vel is None:
                    raise ValueError("follow_object_xy requires object_pos, ee_pos, and object_lin_vel")
                relative_xy = object_pos[:, :2] - ee_pos[:, :2]
                new_hold = active & self.count.eq(0)
                self.anchor_object_to_ee_xy[new_hold] = relative_xy[new_hold]
                velocity_xy = object_lin_vel[:, :2].clamp(-self.max_follow_speed, self.max_follow_speed)
                desired_delta_xy = (
                    velocity_xy * self.control_dt
                    + self.follow_gain * (relative_xy - self.anchor_object_to_ee_xy)
                )
                gated[active, :2] = (desired_delta_xy[active] / self.action_scale).clamp(-1.0, 1.0)

        self.count = torch.where(active, self.count + 1, self.count)
        self.ready |= self.count >= self.hold_steps
        return gated, active


def make_grasp_hold_action_gate(
    hold_steps: int,
    max_upward_dz: float = 0.0,
    force_close: bool = True,
    follow_object_xy: bool = False,
    control_dt: float = 0.02,
    action_scale: float = 0.5,
    follow_gain: float = 0.5,
    max_follow_speed: float = 0.15,
    freeze_rotation: bool = False,
):
    """Build the TorchRL inverse transform that applies :class:`GraspHoldGate`."""
    from torchrl.envs import Transform

    class GraspHoldActionGate(Transform):
        def __init__(self):
            super().__init__(in_keys=[], out_keys=[])
            self.controller = GraspHoldGate(
                hold_steps, max_upward_dz, force_close, follow_object_xy, control_dt,
                action_scale, follow_gain, max_follow_speed, freeze_rotation,
            )

        def _inv_call(self, tensordict):
            action = tensordict.get("action")
            grasped = tensordict.get(("privileged", "is_grasped"))
            kwargs = {}
            if self.controller.follow_object_xy:
                kwargs = {
                    "object_pos": tensordict.get(("privileged", "food_pos")),
                    "ee_pos": tensordict.get(("proprio", "ee_pos")),
                    "object_lin_vel": tensordict.get(("privileged", "food_lin_vel")),
                }
            action, _ = self.controller(action, grasped, **kwargs)
            tensordict.set("action", action)
            return tensordict

        def _reset(self, tensordict, tensordict_reset):
            mask = None if tensordict is None else tensordict.get("_reset", None)
            self.controller.reset(mask)
            return tensordict_reset

        def _reset_on_native_autoreset(self, tensordict, tensordict_reset):
            # IsaacLabWrapper resets finished rows inside step(); TorchRL uses
            # this distinct hook instead of _reset for that path.
            return self._reset(tensordict, tensordict_reset)

    return GraspHoldActionGate()


class GeometricMicroLiftOption:
    """Position-only close-to-lift option for the speed-free policy.

    The option arms after a geometrically aligned, partially obstructed close
    has persisted for ``close_steps``.  It then keeps the fingers closed,
    holds the object-to-TCP XY offset with proportional visual servoing, and
    tracks a smooth vertical TCP target.  Its inputs are deliberately limited
    to current relative geometry, TCP position, and finger encoders: there is
    no object velocity, contact, force, tactile, current, or torque signal.
    """

    def __init__(
        self,
        close_steps: int,
        lift_steps: int,
        hold_steps: int,
        lift_height: float,
        distance_threshold: float,
        min_close_fraction: float,
        max_close_fraction: float,
        open_pos: float,
        closed_pos: float,
        action_scale: float,
        xy_gain: float = 0.5,
        z_gain: float = 1.0,
        max_xy_step: float = 0.002,
        max_z_step: float = 0.001,
        max_finger_delta: float = 0.00025,
        force_close: bool = True,
        freeze_rotation: bool = True,
    ):
        if close_steps < 1 or lift_steps < 1 or hold_steps < 0:
            raise ValueError("close_steps/lift_steps must be >= 1 and hold_steps must be >= 0")
        if lift_height <= 0 or distance_threshold <= 0 or action_scale <= 0:
            raise ValueError("lift_height/distance_threshold/action_scale must be positive")
        if not 0.0 <= min_close_fraction < max_close_fraction <= 1.0:
            raise ValueError("close fractions must satisfy 0 <= min < max <= 1")
        if abs(open_pos - closed_pos) <= 1.0e-9:
            raise ValueError("open_pos and closed_pos must differ")
        if xy_gain < 0 or z_gain <= 0 or max_xy_step <= 0 or max_z_step <= 0 or max_finger_delta < 0:
            raise ValueError("gains/step limits must be positive (xy_gain may be zero)")
        self.close_steps = int(close_steps)
        self.lift_steps = int(lift_steps)
        self.hold_steps = int(hold_steps)
        self.lift_height = float(lift_height)
        self.distance_threshold = float(distance_threshold)
        self.min_close_fraction = float(min_close_fraction)
        self.max_close_fraction = float(max_close_fraction)
        self.open_pos = float(open_pos)
        self.closed_pos = float(closed_pos)
        self.action_scale = float(action_scale)
        self.xy_gain = float(xy_gain)
        self.z_gain = float(z_gain)
        self.max_xy_step = float(max_xy_step)
        self.max_z_step = float(max_z_step)
        self.max_finger_delta = float(max_finger_delta)
        self.force_close = bool(force_close)
        self.freeze_rotation = bool(freeze_rotation)
        self.close_count: torch.Tensor | None = None
        self.phase: torch.Tensor | None = None
        self.anchor_offset_xy: torch.Tensor | None = None
        self.anchor_tcp_z: torch.Tensor | None = None
        self.previous_fingers: torch.Tensor | None = None
        self.fingers_initialized: torch.Tensor | None = None

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if self.close_count is None:
            return
        if mask is None:
            self.close_count.zero_()
            self.phase.zero_()
            self.anchor_offset_xy.zero_()
            self.anchor_tcp_z.zero_()
            self.previous_fingers.zero_()
            self.fingers_initialized.zero_()
        else:
            rows = mask.reshape(mask.shape[0], -1).any(-1)
            self.close_count[rows] = 0
            self.phase[rows] = 0
            self.anchor_offset_xy[rows] = 0.0
            self.anchor_tcp_z[rows] = 0.0
            self.previous_fingers[rows] = 0.0
            self.fingers_initialized[rows] = False

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        return x.square() * (3.0 - 2.0 * x)

    def __call__(
        self,
        action: torch.Tensor,
        object_to_tcp: torch.Tensor,
        ee_pos: torch.Tensor,
        finger_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action.shape[-1] != 7:
            raise ValueError(f"geometric micro-lift requires 7-D ee_delta_pose actions, got {tuple(action.shape)}")
        n = action.shape[0]
        if object_to_tcp.shape != (n, 3) or ee_pos.shape != (n, 3):
            raise ValueError("object_to_tcp and ee_pos must have shape (N, 3)")
        fingers = finger_pos.reshape(n, -1)
        if fingers.shape[1] < 1:
            raise ValueError("finger_pos must contain at least one encoder value")
        if self.phase is None or self.phase.shape[0] != n or self.phase.device != action.device:
            self.close_count = torch.zeros(n, dtype=torch.long, device=action.device)
            self.phase = torch.zeros(n, dtype=torch.long, device=action.device)
            self.anchor_offset_xy = torch.zeros(n, 2, dtype=action.dtype, device=action.device)
            self.anchor_tcp_z = torch.zeros(n, dtype=action.dtype, device=action.device)
            self.previous_fingers = torch.zeros_like(fingers)
            self.fingers_initialized = torch.zeros(n, dtype=torch.bool, device=action.device)

        span = abs(self.open_pos - self.closed_pos)
        close_fraction = (1.0 - (fingers.mean(-1) - self.closed_pos) / span).clamp(0.0, 1.0)
        finger_delta = torch.amax(torch.abs(fingers - self.previous_fingers), dim=-1)
        fingers_stable = self.fingers_initialized & (finger_delta <= self.max_finger_delta)
        aligned_close = (
            (torch.linalg.vector_norm(object_to_tcp, dim=-1) <= self.distance_threshold)
            & (close_fraction >= self.min_close_fraction)
            & (close_fraction <= self.max_close_fraction)
            & fingers_stable
        )
        inactive = self.phase.eq(0)
        self.close_count = torch.where(inactive & aligned_close, self.close_count + 1, torch.zeros_like(self.close_count))
        trigger = inactive & (self.close_count >= self.close_steps)
        self.phase[trigger] = 1
        self.anchor_offset_xy[trigger] = object_to_tcp[trigger, :2]
        self.anchor_tcp_z[trigger] = ee_pos[trigger, 2]

        total_steps = self.lift_steps + self.hold_steps
        active = self.phase.gt(0) & self.phase.le(total_steps)
        out = action.clone()
        if bool(active.any()):
            if self.force_close:
                out[active, -1] = -1.0
            if self.freeze_rotation:
                out[active, 3:6] = 0.0

            xy_error = object_to_tcp[:, :2] - self.anchor_offset_xy
            xy_step = (self.xy_gain * xy_error).clamp(-self.max_xy_step, self.max_xy_step)
            out[active, :2] = (xy_step[active] / self.action_scale).clamp(-1.0, 1.0)

            progress = (self.phase.float() / self.lift_steps).clamp(0.0, 1.0)
            target_z = self.anchor_tcp_z + self.lift_height * self._smoothstep(progress)
            z_step = (self.z_gain * (target_z - ee_pos[:, 2])).clamp(-self.max_z_step, self.max_z_step)
            out[active, 2] = (z_step[active] / self.action_scale).clamp(-1.0, 1.0)

        self.phase = torch.where(active, self.phase + 1, self.phase)
        finished = self.phase > total_steps
        self.phase[finished] = total_steps + 1  # remain latched; never retrigger during this episode
        progress = torch.where(active, (self.phase.float() - 1.0) / self.lift_steps, torch.zeros(n, device=action.device))
        self.previous_fingers.copy_(fingers)
        self.fingers_initialized.fill_(True)
        return out, active, progress.clamp(0.0, 1.0)


def make_geometric_micro_lift_option(**kwargs):
    """Build a TorchRL inverse transform for :class:`GeometricMicroLiftOption`."""
    from torchrl.envs import Transform

    class GeometricMicroLiftActionOption(Transform):
        def __init__(self):
            super().__init__(in_keys=[], out_keys=[])
            self.controller = GeometricMicroLiftOption(**kwargs)

        def _inv_call(self, tensordict):
            history = tensordict.get(("position_history", "position_action"))
            action, _, _ = self.controller(
                tensordict.get("action"),
                history[:, -1, :3],
                tensordict.get(("proprio", "ee_pos")).reshape(history.shape[0], 3),
                tensordict.get(("proprio", "gripper_pos")),
            )
            tensordict.set("action", action)
            return tensordict

        def _reset(self, tensordict, tensordict_reset):
            mask = None if tensordict is None else tensordict.get("_reset", None)
            self.controller.reset(mask)
            return tensordict_reset

        def _reset_on_native_autoreset(self, tensordict, tensordict_reset):
            return self._reset(tensordict, tensordict_reset)

    return GeometricMicroLiftActionOption()


class GraspPoseCloseRetractOption:
    """GAP-style pose-triggered close then vertical retract primitive.

    It consumes only a candidate pose error and TCP pose.  The trigger and
    controller are therefore identical in simulation and on hardware once a
    depth grasp detector supplies candidate poses; no speed/contact/force
    quantity is read.
    """

    def __init__(
        self,
        translation_threshold: float,
        rotation_threshold: float,
        approach_steps: int,
        close_steps: int,
        retract_steps: int,
        retract_height: float,
        action_scale: float,
        close_action: float = -0.15,
        position_gain: float = 0.5,
        rotation_gain: float = 0.5,
        max_position_step: float = 0.002,
        max_rotation_step: float = 0.01,
    ):
        if translation_threshold <= 0 or rotation_threshold <= 0 or action_scale <= 0:
            raise ValueError("pose thresholds and action_scale must be positive")
        if approach_steps < 1 or close_steps < 1 or retract_steps < 1 or retract_height <= 0:
            raise ValueError("approach/close/retract steps and retract_height must be positive")
        if position_gain <= 0 or rotation_gain <= 0 or max_position_step <= 0 or max_rotation_step <= 0:
            raise ValueError("pose option gains and step limits must be positive")
        if not -1.0 <= close_action < 1.0:
            raise ValueError("close_action must be in [-1, 1)")
        self.translation_threshold = float(translation_threshold)
        self.rotation_threshold = float(rotation_threshold)
        self.approach_steps = int(approach_steps)
        self.close_steps = int(close_steps)
        self.retract_steps = int(retract_steps)
        self.retract_height = float(retract_height)
        self.action_scale = float(action_scale)
        self.close_action = float(close_action)
        self.position_gain = float(position_gain)
        self.rotation_gain = float(rotation_gain)
        self.max_position_step = float(max_position_step)
        self.max_rotation_step = float(max_rotation_step)
        self.phase: torch.Tensor | None = None
        self.anchor_tcp_z: torch.Tensor | None = None

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if self.phase is None:
            return
        if mask is None:
            self.phase.zero_()
            self.anchor_tcp_z.zero_()
        else:
            rows = mask.reshape(mask.shape[0], -1).any(-1)
            self.phase[rows] = 0
            self.anchor_tcp_z[rows] = 0.0

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        return x.square() * (3.0 - 2.0 * x)

    def __call__(
        self, action: torch.Tensor, pose_delta: torch.Tensor, ee_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if action.ndim != 2 or action.shape[-1] != 7:
            raise ValueError("grasp-pose option requires (N, 7) ee_delta_pose+gripper actions")
        if pose_delta.shape != (action.shape[0], 6) or ee_pos.shape != (action.shape[0], 3):
            raise ValueError("pose_delta and ee_pos must be shaped (N,6) and (N,3)")
        n = action.shape[0]
        if self.phase is None or self.phase.shape[0] != n or self.phase.device != action.device:
            self.phase = torch.zeros(n, dtype=torch.long, device=action.device)
            self.anchor_tcp_z = torch.zeros(n, dtype=action.dtype, device=action.device)

        aligned = (
            pose_delta[:, :3].norm(dim=-1) <= self.translation_threshold
        ) & (pose_delta[:, 3:].norm(dim=-1) <= self.rotation_threshold)
        trigger = self.phase.eq(0) & aligned
        self.phase[trigger] = 1
        self.anchor_tcp_z[trigger] = ee_pos[trigger, 2]

        retract_start = self.approach_steps + self.close_steps
        total = retract_start + self.retract_steps
        # Once selected, this option owns the controller until episode reset.
        # After the smooth retract completes it holds the final pose/closure;
        # handing control back immediately made the arm chase the moving ball
        # down again before the physical object could clear 3 cm.
        active = self.phase.gt(0)
        out = action.clone()
        if active.any():
            approaching = active & self.phase.le(self.approach_steps)
            out[approaching, -1] = 1.0
            out[active & ~approaching, -1] = self.close_action
            position_step = (self.position_gain * pose_delta[:, :3]).clamp(
                -self.max_position_step, self.max_position_step
            )
            rotation_step = (self.rotation_gain * pose_delta[:, 3:]).clamp(
                -self.max_rotation_step, self.max_rotation_step
            )
            forward_or_close = active & self.phase.le(retract_start)
            out[forward_or_close, :3] = (position_step[forward_or_close] / self.action_scale).clamp(-1.0, 1.0)
            out[forward_or_close, 3:6] = (rotation_step[forward_or_close] / self.action_scale).clamp(-1.0, 1.0)

            retract = active & self.phase.gt(retract_start)
            if retract.any():
                retract_phase = (self.phase.float() - retract_start) / self.retract_steps
                target_z = self.anchor_tcp_z + self.retract_height * self._smoothstep(retract_phase.clamp(0, 1))
                z_step = (self.position_gain * (target_z - ee_pos[:, 2])).clamp(
                    -self.max_position_step, self.max_position_step
                )
                # Continue visual servoing in XY, but never chase the sphere
                # back down after lift-off.
                out[retract, :2] = (position_step[retract, :2] / self.action_scale).clamp(-1.0, 1.0)
                out[retract, 2] = (z_step[retract] / self.action_scale).clamp(-1.0, 1.0)
                out[retract, 3:6] = 0.0

        advancing = active & self.phase.le(total)
        self.phase = torch.where(advancing, self.phase + 1, self.phase)
        self.phase[self.phase > total] = total + 1  # latched until reset
        return out, active


def make_grasp_pose_close_retract_option(**kwargs):
    """Build the TorchRL inverse transform for the speed-free option."""
    from torchrl.envs import Transform

    class GraspPoseCloseRetractTransform(Transform):
        def __init__(self):
            super().__init__(in_keys=[], out_keys=[])
            self.controller = GraspPoseCloseRetractOption(**kwargs)

        def _inv_call(self, tensordict):
            action, _ = self.controller(
                tensordict.get("action"),
                tensordict.get(("grasp_guidance", "best_pose_delta")),
                tensordict.get(("proprio", "ee_pos")),
            )
            tensordict.set("action", action)
            return tensordict

        def _reset(self, tensordict, tensordict_reset):
            mask = None if tensordict is None else tensordict.get("_reset", None)
            self.controller.reset(mask)
            return tensordict_reset

        def _reset_on_native_autoreset(self, tensordict, tensordict_reset):
            return self._reset(tensordict, tensordict_reset)

    return GraspPoseCloseRetractTransform()


def make_stable_sphere_grasp_option(**kwargs):
    """Build the inverse transform for the validated sphere skill option."""
    from torchrl.envs import Transform

    from pickplace.grasp_skills import (
        SphereCloseHoldLiftOptionConfig,
        StableSphereCloseHoldLiftOption,
    )

    class StableSphereGraspOptionTransform(Transform):
        def __init__(self):
            super().__init__(in_keys=[], out_keys=[])
            self.controller = StableSphereCloseHoldLiftOption(
                SphereCloseHoldLiftOptionConfig(**kwargs)
            )

        def _inv_call(self, tensordict):
            action, _, _ = self.controller(
                tensordict.get("action"),
                tensordict.get(("grasp_guidance", "candidate_pose_deltas")),
                tensordict.get(("proprio", "ee_pos")),
            )
            tensordict.set("action", action)
            return tensordict

        def _reset(self, tensordict, tensordict_reset):
            mask = None if tensordict is None else tensordict.get("_reset", None)
            self.controller.reset(mask)
            return tensordict_reset

        def _reset_on_native_autoreset(self, tensordict, tensordict_reset):
            return self._reset(tensordict, tensordict_reset)

    return StableSphereGraspOptionTransform()


def make_stable_sphere_residual_skill(**kwargs):
    """Build an inverse transform for geometry baseline + bounded PPO residual."""
    from torchrl.envs import Transform

    from pickplace.grasp_skills import (
        SphereGraspSkillConfig,
        StableSphereResidualSkill,
        StableTargetConfig,
    )

    translation_residual_limit = kwargs.pop("translation_residual_limit")
    rotation_residual_limit = kwargs.pop("rotation_residual_limit")
    target_keys = {
        "translation_alpha",
        "rotation_alpha",
        "max_translation_jump",
        "max_rotation_jump",
        "stable_translation",
        "stable_rotation",
        "stable_steps",
        "rotation_weight",
        "switch_margin",
    }
    target_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in target_keys}

    class StableSphereResidualSkillTransform(Transform):
        def __init__(self):
            super().__init__(in_keys=[], out_keys=[])
            self.controller = StableSphereResidualSkill(
                SphereGraspSkillConfig(**kwargs),
                StableTargetConfig(**target_kwargs),
                translation_residual_limit=translation_residual_limit,
                rotation_residual_limit=rotation_residual_limit,
            )

        def _inv_call(self, tensordict):
            action, _, _, _ = self.controller(
                tensordict.get("action"),
                tensordict.get(("grasp_guidance", "candidate_pose_deltas")),
                tensordict.get(("proprio", "ee_pos")),
                tensordict.get(("proprio", "gripper_pos")),
            )
            tensordict.set("action", action)
            return tensordict

        def _reset(self, tensordict, tensordict_reset):
            mask = None if tensordict is None else tensordict.get("_reset", None)
            self.controller.reset(mask)
            return tensordict_reset

        def _reset_on_native_autoreset(self, tensordict, tensordict_reset):
            return self._reset(tensordict, tensordict_reset)

    return StableSphereResidualSkillTransform()


def make_stable_sphere_pick_place_skill(transport_config: dict, **kwargs):
    """Build the G0 grasp-residual + deterministic G1 placement transform."""
    from torchrl.envs import Transform

    from pickplace.grasp_skills import (
        SphereGraspSkillConfig,
        SphereTransportPlaceConfig,
        StableSpherePickPlaceSkill,
        StableSphereResidualSkill,
        StableTargetConfig,
    )

    translation_residual_limit = kwargs.pop("translation_residual_limit")
    rotation_residual_limit = kwargs.pop("rotation_residual_limit")
    target_keys = {
        "translation_alpha",
        "rotation_alpha",
        "max_translation_jump",
        "max_rotation_jump",
        "stable_translation",
        "stable_rotation",
        "stable_steps",
        "rotation_weight",
        "switch_margin",
    }
    target_kwargs = {key: kwargs.pop(key) for key in tuple(kwargs) if key in target_keys}
    action_scale = float(kwargs["action_scale"])

    class StableSpherePickPlaceSkillTransform(Transform):
        def __init__(self):
            super().__init__(in_keys=[], out_keys=[])
            grasp = StableSphereResidualSkill(
                SphereGraspSkillConfig(**kwargs),
                StableTargetConfig(**target_kwargs),
                translation_residual_limit=translation_residual_limit,
                rotation_residual_limit=rotation_residual_limit,
            )
            self.controller = StableSpherePickPlaceSkill(
                grasp,
                SphereTransportPlaceConfig(action_scale=action_scale, **transport_config),
            )

        def _inv_call(self, tensordict):
            action, _, _ = self.controller(
                tensordict.get("action"),
                tensordict.get(("grasp_guidance", "candidate_pose_deltas")),
                tensordict.get(("proprio", "ee_pos")),
                tensordict.get(("proprio", "gripper_pos")),
                tensordict.get(("belt", "bowl_pos")),
            )
            tensordict.set("action", action)
            return tensordict

        def _reset(self, tensordict, tensordict_reset):
            mask = None if tensordict is None else tensordict.get("_reset", None)
            self.controller.reset(mask)
            return tensordict_reset

        def _reset_on_native_autoreset(self, tensordict, tensordict_reset):
            return self._reset(tensordict, tensordict_reset)

    return StableSpherePickPlaceSkillTransform()


class CloseRetractOption:
    """Geometrically triggered close/follow/retract motion primitive.

    Triggering needs only the policy's close command and object/TCP pose.  It
    never reads a grasp, contact, force, tactile, torque, or current signal.
    During the short option the fingers stay closed, XY blends toward object
    velocity following, and Z follows an incremental smoothstep trajectory.
    """

    def __init__(
        self,
        steps: int,
        retract_height: float,
        trigger_distance: float,
        trigger_gripper: float,
        control_dt: float,
        action_scale: float,
        follow_gain: float,
        max_follow_speed: float,
        xy_blend: float,
    ):
        if steps < 1 or retract_height <= 0 or trigger_distance <= 0:
            raise ValueError("steps, retract_height, and trigger_distance must be positive")
        if control_dt <= 0 or action_scale <= 0 or follow_gain < 0 or max_follow_speed <= 0:
            raise ValueError("invalid close-retract controller gain/scale")
        if not 0.0 <= xy_blend <= 1.0:
            raise ValueError("xy_blend must be in [0, 1]")
        self.steps = int(steps)
        self.retract_height = float(retract_height)
        self.trigger_distance = float(trigger_distance)
        self.trigger_gripper = float(trigger_gripper)
        self.control_dt = float(control_dt)
        self.action_scale = float(action_scale)
        self.follow_gain = float(follow_gain)
        self.max_follow_speed = float(max_follow_speed)
        self.xy_blend = float(xy_blend)
        self.phase: torch.Tensor | None = None
        self.latched: torch.Tensor | None = None
        self.anchor_xy: torch.Tensor | None = None

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if self.phase is None:
            return
        if mask is None:
            self.phase.zero_()
            self.latched.zero_()
            self.anchor_xy.zero_()
        else:
            rows = mask.reshape(mask.shape[0], -1).any(-1)
            self.phase[rows] = 0
            self.latched[rows] = False
            self.anchor_xy[rows] = 0.0

    @staticmethod
    def _smoothstep(x: torch.Tensor) -> torch.Tensor:
        return x.square() * (3.0 - 2.0 * x)

    def __call__(
        self,
        action: torch.Tensor,
        object_pos: torch.Tensor,
        ee_pos: torch.Tensor,
        object_lin_vel: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action.shape[-1] != 7:
            raise ValueError(f"close-retract option requires 7-D ee_delta_pose actions, got {tuple(action.shape)}")
        n = action.shape[0]
        if self.phase is None or self.phase.shape[0] != n or self.phase.device != action.device:
            self.phase = torch.zeros(n, dtype=torch.long, device=action.device)
            self.latched = torch.zeros(n, dtype=torch.bool, device=action.device)
            self.anchor_xy = torch.zeros(n, 2, dtype=action.dtype, device=action.device)

        relative = object_pos - ee_pos
        distance = torch.linalg.vector_norm(relative, dim=-1)
        # Re-arm only after an explicit open command or a clearly missed/fallen-away object.
        self.latched &= ~((action[:, -1] > 0.0) | (distance > 2.0 * self.trigger_distance))
        trigger = (
            self.phase.eq(0)
            & ~self.latched
            & (action[:, -1] <= self.trigger_gripper)
            & (distance <= self.trigger_distance)
        )
        self.phase[trigger] = 1
        self.latched[trigger] = True
        self.anchor_xy[trigger] = relative[trigger, :2]
        active = self.phase > 0

        out = action.clone()
        if active.any():
            out[active, -1] = -1.0
            velocity_xy = object_lin_vel[:, :2].clamp(-self.max_follow_speed, self.max_follow_speed)
            desired_xy = velocity_xy * self.control_dt + self.follow_gain * (relative[:, :2] - self.anchor_xy)
            follow_action = (desired_xy / self.action_scale).clamp(-1.0, 1.0)
            out[active, :2] = (
                (1.0 - self.xy_blend) * action[active, :2] + self.xy_blend * follow_action[active]
            )
            phase0 = (self.phase.float() - 1.0) / self.steps
            phase1 = self.phase.float() / self.steps
            dz = self.retract_height * (self._smoothstep(phase1) - self._smoothstep(phase0))
            out[active, 2] = (dz[active] / self.action_scale).clamp(-1.0, 1.0)

        self.phase = torch.where(active, self.phase + 1, self.phase)
        finished = self.phase > self.steps
        self.phase[finished] = 0
        progress = torch.where(active, (self.phase.float() - 1.0) / self.steps, torch.zeros_like(distance))
        return out, active, progress.clamp(0.0, 1.0)


def make_close_retract_option(**kwargs):
    """Build the TorchRL inverse transform for :class:`CloseRetractOption`."""
    from torchrl.envs import Transform

    class CloseRetractActionOption(Transform):
        def __init__(self):
            super().__init__(in_keys=[], out_keys=[])
            self.controller = CloseRetractOption(**kwargs)

        def _inv_call(self, tensordict):
            action, _, _ = self.controller(
                tensordict.get("action"),
                tensordict.get(("privileged", "food_pos")),
                tensordict.get(("proprio", "ee_pos")),
                tensordict.get(("privileged", "food_lin_vel")),
            )
            tensordict.set("action", action)
            return tensordict

        def _reset(self, tensordict, tensordict_reset):
            mask = None if tensordict is None else tensordict.get("_reset", None)
            self.controller.reset(mask)
            return tensordict_reset

        def _reset_on_native_autoreset(self, tensordict, tensordict_reset):
            return self._reset(tensordict, tensordict_reset)

    return CloseRetractActionOption()
