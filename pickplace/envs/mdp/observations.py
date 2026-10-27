# pickplace/envs/mdp/observations.py
"""Observation terms. Positions are expressed in the cell frame (env origin); quaternions are xyzw."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from pickplace.grasp_geometry import (
    deterministic_grasp_points,
    grasp_pose_deltas,
    nearest_grasp_delta,
    quat_xyzw_to_matrix,
    sphere_grasp_frames,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def image_float(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, data_type: str = "rgb") -> torch.Tensor:
    """Raw (unnormalized) camera image cast to float32; values stay in [0, 255] for rgb.

    Isaac Lab's ``ManagerBasedRLEnv`` builds every observation term's gym space as
    ``gym.spaces.Box(..., dtype=np.float32)`` regardless of the term's actual output dtype
    (``manager_based_rl_env.py``). The stock ``isaaclab.envs.mdp.image(normalize=False)`` returns
    the camera sensor's native dtype (``uint8`` for rgb), which makes TorchRL's ``check_env_specs``
    reject the mismatch between the declared float32 spec and the real uint8 tensor (verified on the
    Spark: ``AssertionError: ... Got fake=torch.float32 and real=torch.uint8``). Cast here instead of
    relaxing the spec, per deviation #2 in the plan overview.

    Memory note: a float32 RGB buffer is 4x the size of the native uint8 RGB buffer the camera
    sensor produces (4 bytes/channel vs. 1), and TorchRL stores one copy per transition at both the
    root and "next" keys. See ``docs/environment.md`` for the per-frame memory formula and
    recommended ``num_envs`` ranges when training with cameras on the Spark.
    """
    from isaaclab.envs import mdp as base_mdp

    return base_mdp.image(env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=False).float()


def depth_meters(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "distance_to_image_plane",
    max_distance: float = 3.0,
) -> torch.Tensor:
    """Finite metric depth suitable for storage and real RGB-D replacement.

    Isaac uses ``inf`` for rays that miss all geometry.  Mapping those values
    to the configured far distance keeps every stored frame finite while
    preserving metres as the unit.  Depth remains an observation/label only;
    it is never exposed to the frozen state teacher.
    """
    if max_distance <= 0.0:
        raise ValueError("max_distance must be positive")
    from isaaclab.envs import mdp as base_mdp

    depth = base_mdp.image(
        env, sensor_cfg=sensor_cfg, data_type=data_type, normalize=False
    ).float()
    return torch.nan_to_num(
        depth, nan=max_distance, posinf=max_distance, neginf=0.0
    ).clamp_(0.0, max_distance)


class stacked_image_float(ManagerTermBase):
    """Last ``frame_stack`` camera frames stacked along channels, cast to float32 (values stay in [0, 255]).

    Wraps Isaac Lab's ``stacked_image`` (per-env history, filled with the current frame after a reset) with
    ``normalize=False`` and the float32 cast explained in ``image_float``. Output ``(N, H, W, 3 * frame_stack)``,
    oldest frame first.

    Isaac Lab's observation manager requires class terms to subclass ``ManagerTermBase`` (it checks
    ``issubclass(term_cfg.func, ManagerTermBase)`` before instantiating, and later calls
    ``term_cfg.func.reset()`` only ``if isinstance(term_cfg.func, ManagerTermBase)``), so this wraps
    ``stacked_image`` by subclassing rather than by plain delegation.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        from isaaclab.envs.mdp.observations import stacked_image

        super().__init__(cfg, env)
        self._stacked = stacked_image(cfg, env)

    def reset(self, env_ids: torch.Tensor | None = None):
        self._stacked.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        data_type: str = "rgb",
        frame_stack: int = 1,
    ) -> torch.Tensor:
        return self._stacked(
            env, sensor_cfg=sensor_cfg, data_type=data_type, frame_stack=frame_stack, normalize=False
        ).float()


def asset_pos_cell(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_pos_w.torch - env.scene.env_origins


def asset_quat_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_quat_w.torch.clone()


def asset_lin_vel_cell(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-aligned linear velocity; env origins are translational only, so no frame rotation is needed."""
    return env.scene[asset_cfg.name].data.root_lin_vel_w.torch.clone()


def conveyor_speed(
    env: ManagerBasedEnv,
    belt_event_name: str = "reset_belt",
) -> torch.Tensor:
    """Per-environment conveyor speed sampled by the reset event, shaped ``(N, 1)``."""
    event = env.event_manager.get_term_cfg(belt_event_name).func
    return event.speed.unsqueeze(-1).clone()


def ee_pos_cell(env: ManagerBasedEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    return env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :] - env.scene.env_origins


def ee_quat_w(env: ManagerBasedEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    return env.scene[ee_frame_cfg.name].data.target_quat_w.torch[:, 0, :].clone()


def gripper_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids].clone()


def gripper_close_fraction(
    positions: torch.Tensor, open_pos: float, closed_pos: float
) -> torch.Tensor:
    """Mean normalized closure, independent of the joint's sign convention."""
    delta = float(closed_pos) - float(open_pos)
    if abs(delta) < 1.0e-6:
        raise ValueError("gripper open_pos and closed_pos must differ")
    return ((positions.mean(-1) - float(open_pos)) / delta).clamp(0.0, 1.0)


def gripper_blocked_between_limits(
    positions: torch.Tensor, open_pos: float, closed_pos: float, margin: float
) -> torch.Tensor:
    """Whether every finger stopped strictly between open and closed limits."""
    lower, upper = sorted((float(open_pos), float(closed_pos)))
    return ((positions > lower + float(margin)) & (positions < upper - float(margin))).all(dim=-1)


class motion_history(ManagerTermBase):
    """Finite state/action history for dynamic interception, with no contact signal.

    Each frame contains, in order: object-to-TCP position (3), object linear
    velocity (3), finite-difference TCP velocity (3), finger positions, belt
    speed (1), and the previous policy action.  All quantities have direct
    camera/encoder/controller equivalents on the real cell.  In particular,
    neither contact state nor force, torque, tactile, or motor-current data is
    used.

    The oldest frame is first.  On reset the current frame is repeated so a
    policy never sees an artificial zero transient at the start of an episode.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        history_steps = int(params["history_steps"])
        if history_steps < 2:
            raise ValueError(f"history_steps must be >= 2, got {history_steps}")
        robot_cfg = params["robot_cfg"]
        finger_count = len(robot_cfg.joint_ids)
        action_dim = int(env.action_manager.action.shape[-1])
        self.history_steps = history_steps
        self.feature_dim = 10 + finger_count + action_dim
        self.history = torch.zeros(
            env.num_envs, history_steps, self.feature_dim, dtype=torch.float32, device=env.device
        )
        self.previous_ee = torch.zeros(env.num_envs, 3, dtype=torch.float32, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.history.zero_()
            self.previous_ee.zero_()
            self.initialized.zero_()
        else:
            self.history[env_ids] = 0.0
            self.previous_ee[env_ids] = 0.0
            self.initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        history_steps: int,
        robot_cfg: SceneEntityCfg,
        belt_event_name: str = "reset_belt",
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if int(history_steps) != self.history_steps:
            raise ValueError("history_steps cannot change after motion_history is constructed")
        food_pos = asset_pos_cell(env, food_cfg)
        food_vel = asset_lin_vel_cell(env, food_cfg)
        ee_pos = ee_pos_cell(env, ee_frame_cfg)
        dt = float(env.step_dt)
        ee_vel = (ee_pos - self.previous_ee) / dt
        ee_vel = torch.where(self.initialized[:, None], ee_vel, torch.zeros_like(ee_vel))
        fingers = gripper_pos(env, robot_cfg)
        speed = conveyor_speed(env, belt_event_name)
        last_action = env.action_manager.prev_action.clone()
        frame = torch.cat((food_pos - ee_pos, food_vel, ee_vel, fingers, speed, last_action), dim=-1)

        fresh = ~self.initialized
        if fresh.any():
            self.history[fresh] = frame[fresh, None, :]
        continuing = self.initialized
        if continuing.any():
            self.history[continuing, :-1] = self.history[continuing, 1:].clone()
            self.history[continuing, -1] = frame[continuing]
        self.previous_ee.copy_(ee_pos)
        self.initialized.fill_(True)
        return self.history.clone()


class position_action_history(ManagerTermBase):
    """Position-only temporal context for speed-free dynamic manipulation.

    Each oldest-to-newest frame contains object-to-TCP position (3), finger
    positions, and the previous policy action.  Deliberately absent are object,
    conveyor, joint, and finite-difference TCP velocities.  A recurrent policy
    may learn temporal correspondence from positions, matching what a real
    vision/encoder stack can provide without a dedicated speed estimator.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        history_steps = int(params["history_steps"])
        if history_steps < 2:
            raise ValueError(f"history_steps must be >= 2, got {history_steps}")
        finger_count = len(params["robot_cfg"].joint_ids)
        action_dim = int(env.action_manager.action.shape[-1])
        self.history_steps = history_steps
        self.feature_dim = 3 + finger_count + action_dim
        self.history = torch.zeros(
            env.num_envs, history_steps, self.feature_dim, dtype=torch.float32, device=env.device
        )
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.history.zero_()
            self.initialized.zero_()
        else:
            self.history[env_ids] = 0.0
            self.initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        history_steps: int,
        robot_cfg: SceneEntityCfg,
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if int(history_steps) != self.history_steps:
            raise ValueError("history_steps cannot change after position_action_history is constructed")
        relative_position = asset_pos_cell(env, food_cfg) - ee_pos_cell(env, ee_frame_cfg)
        fingers = gripper_pos(env, robot_cfg)
        last_action = env.action_manager.prev_action.clone()
        frame = torch.cat((relative_position, fingers, last_action), dim=-1)

        fresh = ~self.initialized
        if fresh.any():
            self.history[fresh] = frame[fresh, None, :]
        continuing = self.initialized
        if continuing.any():
            self.history[continuing, :-1] = self.history[continuing, 1:].clone()
            self.history[continuing, -1] = frame[continuing]
        self.initialized.fill_(True)
        return self.history.clone()


class grasp_candidate_points_ee(ManagerTermBase):
    """GAP-RL-style analytic grasp candidates represented in the TCP frame.

    Four top-down grasp regions are represented by deterministic anisotropic
    Gaussian point sets.  Simulation obtains the region centre from object
    position; a real deployment replaces this term with candidates from its
    depth-based grasp detector.  The returned tensor contains positions only
    and has shape ``(num_envs, num_candidates, points_per_candidate, 3)``.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        params = cfg.params
        candidates = int(params.get("num_candidates", 4))
        points = int(params.get("points_per_candidate", 16))
        if candidates < 1 or points < 2:
            raise ValueError("grasp candidates must be >= 1 and points_per_candidate >= 2")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(params.get("template_seed", 0)))
        standard = torch.randn(candidates, points, 3, generator=generator, dtype=torch.float32)
        std = torch.tensor(params.get("point_std", (0.025, 0.008, 0.006)), dtype=torch.float32)
        local = standard * std
        yaw = torch.arange(candidates, dtype=torch.float32) * (torch.pi / candidates)
        cosine, sine = yaw.cos()[:, None], yaw.sin()[:, None]
        x, y = local[..., 0].clone(), local[..., 1].clone()
        local[..., 0] = cosine * x - sine * y
        local[..., 1] = sine * x + cosine * y
        self.template = local.to(env.device)
        self.num_candidates = candidates
        self.points_per_candidate = points

    def reset(self, env_ids: torch.Tensor | None = None):
        return None

    def __call__(
        self,
        env: ManagerBasedEnv,
        num_candidates: int = 4,
        points_per_candidate: int = 16,
        template_seed: int = 0,
        point_std: tuple[float, float, float] = (0.025, 0.008, 0.006),
        grasp_height: float = 0.0,
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        del template_seed, point_std
        if int(num_candidates) != self.num_candidates or int(points_per_candidate) != self.points_per_candidate:
            raise ValueError("grasp candidate shape cannot change after construction")
        from isaaclab.utils.math import quat_apply_inverse

        centre = asset_pos_cell(env, food_cfg).clone()
        centre[:, 2] += float(grasp_height)
        points_cell = centre[:, None, None, :] + self.template[None, :, :, :]
        tcp_pos = ee_pos_cell(env, ee_frame_cfg)
        tcp_quat = ee_quat_w(env, ee_frame_cfg)
        relative = points_cell - tcp_pos[:, None, None, :]
        quat = tcp_quat[:, None, None, :].expand(-1, self.num_candidates, self.points_per_candidate, -1)
        return quat_apply_inverse(quat.reshape(-1, 4), relative.reshape(-1, 3)).reshape_as(relative)


class sphere_grasp_candidate_points_ee(ManagerTermBase):
    """True 6D sphere grasp candidates encoded as 40x20 TCP-frame points."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.num_candidates = int(p.get("num_candidates", 40))
        self.points_per_candidate = int(p.get("points_per_candidate", 20))
        if self.num_candidates != 40 or self.points_per_candidate != 20:
            raise ValueError("sphere_gap uses exactly 40 candidates x 20 points")

    def reset(self, env_ids: torch.Tensor | None = None):
        return None

    def __call__(
        self,
        env: ManagerBasedEnv,
        num_candidates: int = 40,
        points_per_candidate: int = 20,
        tcp_height_offset: float = 0.002,
        canonical_yaw: float | None = None,
        grasp_width: float = 0.044,
        food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        if int(num_candidates) != self.num_candidates or int(points_per_candidate) != self.points_per_candidate:
            raise ValueError("sphere grasp candidate shape cannot change after construction")
        centres = asset_pos_cell(env, food_cfg)
        positions, rotations = sphere_grasp_frames(
            centres,
            num_candidates=self.num_candidates,
            tcp_height_offset=tcp_height_offset,
            canonical_yaw=canonical_yaw,
        )
        points_cell = deterministic_grasp_points(
            positions, rotations, points_per_candidate=self.points_per_candidate, grasp_width=grasp_width
        )
        tcp_pos = ee_pos_cell(env, ee_frame_cfg)
        tcp_rot = quat_xyzw_to_matrix(ee_quat_w(env, ee_frame_cfg))
        relative = points_cell - tcp_pos[:, None, None, :]
        return torch.einsum("nij,ncpj->ncpi", tcp_rot.transpose(-1, -2), relative)


def sphere_best_grasp_pose_delta(
    env: ManagerBasedEnv,
    num_candidates: int = 40,
    tcp_height_offset: float = 0.002,
    canonical_yaw: float | None = None,
    rotation_weight: float = 0.05,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Nearest candidate error: cell-frame translation + world-frame rotvec.

    This is geometric and deployable: on hardware the candidate poses come
    from the depth grasp detector and the TCP pose from robot FK.  It contains
    no object/conveyor velocity, contact, force, torque, current, or tactile
    signal.
    """
    positions, rotations = sphere_grasp_frames(
        asset_pos_cell(env, food_cfg),
        num_candidates=int(num_candidates),
        tcp_height_offset=float(tcp_height_offset),
        canonical_yaw=canonical_yaw,
    )
    translation, rotation, _ = nearest_grasp_delta(
        ee_pos_cell(env, ee_frame_cfg),
        ee_quat_w(env, ee_frame_cfg),
        positions,
        rotations,
        rotation_weight=float(rotation_weight),
    )
    return torch.cat((translation, rotation), dim=-1)


def sphere_grasp_candidate_pose_deltas(
    env: ManagerBasedEnv,
    num_candidates: int = 40,
    tcp_height_offset: float = 0.002,
    canonical_yaw: float | None = None,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """All 6D candidate errors for cross-frame region identity tracking."""
    positions, rotations = sphere_grasp_frames(
        asset_pos_cell(env, food_cfg),
        num_candidates=int(num_candidates),
        tcp_height_offset=float(tcp_height_offset),
        canonical_yaw=canonical_yaw,
    )
    translation, rotation = grasp_pose_deltas(
        ee_pos_cell(env, ee_frame_cfg),
        ee_quat_w(env, ee_frame_cfg),
        positions,
        rotations,
    )
    return torch.cat((translation, rotation), dim=-1)


def dual_finger_contact_mask(
    env: ManagerBasedEnv,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_finger_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_finger_contact"),
    force_threshold: float = 0.05,
) -> torch.Tensor:
    """Training-only label: both fingers contact the filtered food body."""
    masks = []
    for cfg in (left_sensor_cfg, right_sensor_cfg):
        forces = env.scene[cfg.name].data.force_matrix_w
        if forces is None:
            raise RuntimeError(f"{cfg.name} must set filter_prim_paths_expr to expose force_matrix_w")
        masks.append(forces.norm(dim=-1).amax(dim=(-1, -2)) > float(force_threshold))
    return masks[0] & masks[1]


def grasped_mask(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    distance_threshold: float = 0.04,
    open_pos: float = 0.04,
    closed_pos: float = 0.0,
    margin: float = 0.004,
) -> torch.Tensor:
    """Food near the TCP while fingers stopped between open and closed (i.e. blocked by the food)."""
    food_pos = env.scene[food_cfg.name].data.root_pos_w.torch
    tcp_pos = env.scene[ee_frame_cfg.name].data.target_pos_w.torch[:, 0, :]
    near = torch.linalg.vector_norm(food_pos - tcp_pos, dim=-1) < distance_threshold
    fingers = env.scene[robot_cfg.name].data.joint_pos.torch[:, robot_cfg.joint_ids]
    blocked = gripper_blocked_between_limits(fingers, open_pos, closed_pos, margin)
    return near & blocked


def is_grasped(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    distance_threshold: float = 0.04,
    open_pos: float = 0.04,
    closed_pos: float = 0.0,
    margin: float = 0.004,
) -> torch.Tensor:
    mask = grasped_mask(env, robot_cfg, food_cfg, ee_frame_cfg, distance_threshold, open_pos, closed_pos, margin)
    return mask.float().unsqueeze(-1)


def kinematic_grasp_proxy(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg,
    food_cfg: SceneEntityCfg = SceneEntityCfg("food"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    distance_threshold: float = 0.04,
    open_pos: float = 0.04,
    closed_pos: float = 0.0,
    margin: float = 0.004,
) -> torch.Tensor:
    """Deployable grasp proxy from pose and finger aperture only.

    The formula intentionally matches the historical teacher's phase bit so
    its weights can be transferred, but it queries no simulator contacts.
    On hardware the inputs come from visual object tracking, robot FK, and the
    gripper position encoder; no force, tactile, torque, or current is needed.
    """
    return is_grasped(
        env, robot_cfg, food_cfg, ee_frame_cfg, distance_threshold,
        open_pos, closed_pos, margin,
    )
