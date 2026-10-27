"""True closed-loop MuJoCo bridge for JAKA ZU5 + AG95 moving-sphere J0."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mujoco_sim.env import POLICY_DT, ROOT, StepResult, xyzw_from_wxyz
from mujoco_sim.jaka_scene import (
    build_jaka_multi_sphere_scene_xml,
    build_jaka_queued_sphere_scene_xml,
    build_jaka_scene_xml,
    build_jaka_static_sphere_moving_bowl_queue_scene_xml,
    build_jaka_static_sphere_moving_bowl_scene_xml,
)
from mujoco_sim.moving_sphere import (
    ACTION_SCALE,
    BELT_SPEED,
    MAX_SECONDS,
    SPHERE_START_Z,
    SphereDiagnostics,
    moving_sphere_outcome,
    moving_sphere_place_outcome,
)
from pickplace.grasp_geometry import (
    deterministic_grasp_points,
    grasp_pose_deltas,
    nearest_grasp_delta,
    quat_xyzw_to_matrix,
    sphere_grasp_frames,
)
from pickplace.grasp_skills import (
    SphereGraspSkillConfig,
    SphereTransportPlaceConfig,
    StableSphereGraspSkill,
    StableSpherePickPlaceSkill,
    StableSphereResidualSkill,
    StableTargetConfig,
)


JAKA_DEFAULT_Q = np.asarray(
    [-2.341213, 1.740149, 1.777842, 1.194395, 4.712393, -2.341218], dtype=float
)
JAKA_JOINT_LOW = np.asarray([-6.28, -1.48, -3.05, -1.48, -6.28, -6.28], dtype=float)
JAKA_JOINT_HIGH = np.asarray([6.28, 4.62, 3.05, 4.62, 6.28, 6.28], dtype=float)
JAKA_JOINT_VELOCITY = np.full(6, 3.14, dtype=float)
JAKA_HOME_TCP = np.asarray([0.45, 0.30, 0.134], dtype=float)
AG95_OPEN = 0.0
AG95_CLOSED = 0.943


class JakaMovingSphereMuJoCo:
    """J0 state-policy environment driven by MuJoCo state at every step."""

    def __init__(
        self,
        *,
        seed: int = 31,
        belt_speed: float = BELT_SPEED,
        grasp_assist: bool = False,
        max_seconds: float = MAX_SECONDS,
        sphere_start_x: float = 0.20,
        sphere_start_y: float = 0.30,
        controller_profile: str = "j0_s1",
        pick_place: bool = False,
        destination_bowl_pos: tuple[float, float, float] = (0.45, 0.0, 0.0),
        translation_residual_limit: float | None = None,
        rotation_residual_limit: float | None = None,
        task_variant: str = "moving_sphere_static_bowl",
        num_spheres: int = 1,
        sphere_queue_spacing: float | None = None,
        moving_bowl_queue_spacing: float | None = None,
    ):
        if task_variant not in (
            "moving_sphere_static_bowl",
            "static_sphere_moving_bowl",
        ):
            raise ValueError("unsupported JAKA task_variant")
        self.num_spheres = int(num_spheres)
        if not 1 <= self.num_spheres <= 32:
            raise ValueError("num_spheres must be between 1 and 32")
        self.queued_spheres = sphere_queue_spacing is not None
        self.queued_moving_bowls = moving_bowl_queue_spacing is not None
        if self.queued_spheres and self.queued_moving_bowls:
            raise ValueError("only one conveyor queue mode may be enabled")
        self.sphere_queue_spacing = (
            None if sphere_queue_spacing is None else float(sphere_queue_spacing)
        )
        self.moving_bowl_queue_spacing = (
            None if moving_bowl_queue_spacing is None else float(moving_bowl_queue_spacing)
        )
        if self.queued_spheres:
            if task_variant != "moving_sphere_static_bowl" or self.num_spheres <= 1:
                raise ValueError(
                    "sphere_queue_spacing requires multiple moving spheres and a static bowl"
                )
            if self.sphere_queue_spacing is None or self.sphere_queue_spacing <= 0.0:
                raise ValueError("sphere_queue_spacing must be positive")
        if self.queued_moving_bowls:
            if task_variant != "static_sphere_moving_bowl" or self.num_spheres <= 1:
                raise ValueError(
                    "moving_bowl_queue_spacing requires multiple static spheres"
                )
            if self.num_spheres > 10:
                raise ValueError("the static supply rack currently supports at most 10 spheres")
            if self.moving_bowl_queue_spacing is None or self.moving_bowl_queue_spacing <= 0.0:
                raise ValueError("moving_bowl_queue_spacing must be positive")
        if (
            self.num_spheres > 1
            and task_variant == "static_sphere_moving_bowl"
            and not self.queued_moving_bowls
        ):
            raise ValueError("multiple static spheres require a moving-bowl queue spacing")
        self.queued_carriers = self.queued_spheres or self.queued_moving_bowls
        self.task_variant = task_variant
        if self.queued_moving_bowls:
            assert self.moving_bowl_queue_spacing is not None
            scene_xml = build_jaka_static_sphere_moving_bowl_queue_scene_xml(
                self.num_spheres, self.moving_bowl_queue_spacing
            )
        elif task_variant == "static_sphere_moving_bowl":
            scene_xml = build_jaka_static_sphere_moving_bowl_scene_xml()
        elif self.queued_spheres:
            assert self.sphere_queue_spacing is not None
            scene_xml = build_jaka_queued_sphere_scene_xml(
                self.num_spheres, self.sphere_queue_spacing
            )
        else:
            scene_xml = build_jaka_multi_sphere_scene_xml(self.num_spheres)
        self.model = mujoco.MjModel.from_xml_string(scene_xml)
        self.data = mujoco.MjData(self.model)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.belt_speed = float(belt_speed)
        self.grasp_assist = bool(grasp_assist)
        self.max_seconds = float(max_seconds)
        self.sphere_start_x = float(sphere_start_x)
        self.sphere_start_y = float(sphere_start_y)
        if controller_profile not in ("j0_s1", "j0_s2", "j1", "static_pick"):
            raise ValueError("controller_profile must be j0_s1, j0_s2, j1, or static_pick")
        self.controller_profile = controller_profile
        self.pick_place = bool(
            pick_place
            or controller_profile == "j1"
            or task_variant == "static_sphere_moving_bowl"
        )
        self.destination_bowl_pos = np.asarray(destination_bowl_pos, dtype=float)
        if self.destination_bowl_pos.shape != (3,):
            raise ValueError("destination_bowl_pos must contain three values")
        destination_name = (
            "moving_bowl"
            if task_variant == "static_sphere_moving_bowl"
            else "destination_bowl"
        )
        self.destination_bowl_body = self.model.body(destination_name).id
        if task_variant == "moving_sphere_static_bowl":
            self.model.body_pos[self.destination_bowl_body] = self.destination_bowl_pos
        else:
            self.supply_bowl_body = self.model.body("supply_bowl").id
            # A 24 mm physical riser keeps the sphere centre near 5 cm, above
            # the ZU5's low-wrist singular region while retaining real bowl
            # support/contact.  This is part of the JAKA work-cell geometry,
            # not a success-threshold relaxation.
            self.model.body_pos[self.supply_bowl_body] = np.asarray((0.45, 0.00, 0.024))
        self.physics_steps = round(POLICY_DT / self.model.opt.timestep)
        if not np.isclose(self.physics_steps * self.model.opt.timestep, POLICY_DT):
            raise ValueError("MuJoCo timestep must divide the 0.02 s policy step")

        self.arm_qpos = np.asarray(
            [self.model.joint(f"0_joint_{index}").qposadr[0] for index in range(1, 7)]
        )
        self.arm_dof = np.asarray(
            [self.model.joint(f"0_joint_{index}").dofadr[0] for index in range(1, 7)]
        )
        linkage_names = (
            "0_left_inner_knuckle_joint",
            "0_left_outer_knuckle_joint",
            "0_left_finger_joint",
            "0_right_inner_knuckle_joint",
            "0_right_outer_knuckle_joint",
            "0_right_finger_joint",
        )
        self.linkage_qpos = np.asarray([self.model.joint(name).qposadr[0] for name in linkage_names])
        self.linkage_dof = np.asarray([self.model.joint(name).dofadr[0] for name in linkage_names])
        self.finger_qpos = np.asarray(
            [self.model.joint("0_left_outer_knuckle_joint").qposadr[0]]
        )
        self.finger_dof = np.asarray(
            [self.model.joint("0_left_outer_knuckle_joint").dofadr[0]]
        )
        self.robot_qpos = self.arm_qpos
        pallet_joint_names = [
            "pallet_slide",
            *(f"pallet_slide_{index}" for index in range(1, self.num_spheres)),
        ] if self.queued_carriers else ["pallet_slide"]
        pallet_actuator_names = [
            "pallet_drive",
            *(f"pallet_drive_{index}" for index in range(1, self.num_spheres)),
        ] if self.queued_carriers else ["pallet_drive"]
        self.pallet_qpos_all = np.asarray(
            [self.model.joint(name).qposadr[0] for name in pallet_joint_names], dtype=int
        )
        self.pallet_dof_all = np.asarray(
            [self.model.joint(name).dofadr[0] for name in pallet_joint_names], dtype=int
        )
        self.pallet_act_all = np.asarray(
            [self.model.actuator(name).id for name in pallet_actuator_names], dtype=int
        )
        self.pallet_qpos = int(self.pallet_qpos_all[0])
        self.pallet_dof = int(self.pallet_dof_all[0])
        if self.task_variant == "static_sphere_moving_bowl":
            moving_bowl_joint_names = [
                "moving_bowl_free",
                *(
                    f"moving_bowl_free_{index}"
                    for index in range(1, self.num_spheres)
                ),
            ] if self.queued_moving_bowls else ["moving_bowl_free"]
            moving_bowl_body_names = [
                "moving_bowl",
                *(f"moving_bowl_{index}" for index in range(1, self.num_spheres)),
            ] if self.queued_moving_bowls else ["moving_bowl"]
            self.moving_bowl_qpos_all = np.asarray(
                [self.model.joint(name).qposadr[0] for name in moving_bowl_joint_names],
                dtype=int,
            )
            self.moving_bowl_dof_all = np.asarray(
                [self.model.joint(name).dofadr[0] for name in moving_bowl_joint_names],
                dtype=int,
            )
            self.moving_bowl_body_all = np.asarray(
                [self.model.body(name).id for name in moving_bowl_body_names], dtype=int
            )
            self.moving_bowl_qpos = int(self.moving_bowl_qpos_all[0])
            self.moving_bowl_dof = int(self.moving_bowl_dof_all[0])
            self.destination_bowl_body = int(self.moving_bowl_body_all[0])
        else:
            self.moving_bowl_qpos_all = np.asarray([], dtype=int)
            self.moving_bowl_dof_all = np.asarray([], dtype=int)
            self.moving_bowl_body_all = np.asarray([], dtype=int)
            self.moving_bowl_qpos = None
            self.moving_bowl_dof = None
        food_names = ["food", *(f"food_{index}" for index in range(1, self.num_spheres))]
        food_joint_names = [
            "food_free", *(f"food_free_{index}" for index in range(1, self.num_spheres))
        ]
        food_geom_names = [
            "food_geom", *(f"food_geom_{index}" for index in range(1, self.num_spheres))
        ]
        self.food_qpos_all = np.asarray(
            [self.model.joint(name).qposadr[0] for name in food_joint_names], dtype=int
        )
        self.food_dof_all = np.asarray(
            [self.model.joint(name).dofadr[0] for name in food_joint_names], dtype=int
        )
        self.food_body_all = np.asarray(
            [self.model.body(name).id for name in food_names], dtype=int
        )
        self.food_geom_all = np.asarray(
            [self.model.geom(name).id for name in food_geom_names], dtype=int
        )
        self.arm_act = np.asarray([self.model.actuator(f"arm{index}").id for index in range(1, 7)])
        self.finger_act = self.model.actuator("fingers_actuator").id
        self.pallet_act = int(self.pallet_act_all[0])
        self.tcp_site = self.model.site("0_grip_site").id
        self.active_sphere_index = 0
        self.food_qpos = int(self.food_qpos_all[0])
        self.food_dof = int(self.food_dof_all[0])
        self.food_body = int(self.food_body_all[0])
        self.food_geom = int(self.food_geom_all[0])
        self.left_finger_body = self.model.body("0_left_finger").id
        self.right_finger_body = self.model.body("0_right_finger").id
        self.food_contype_all = self.model.geom_contype[self.food_geom_all].astype(int).copy()
        self.food_conaffinity_all = self.model.geom_conaffinity[self.food_geom_all].astype(int).copy()
        self.food_contype = int(self.food_contype_all[0])
        self.food_conaffinity = int(self.food_conaffinity_all[0])

        skill_kwargs = {
            "action_scale": ACTION_SCALE,
            "intercept_x": 0.45,
            "intercept_lead_distance": 0.08,
            "hover_height": 0.08,
            "hover_translation_threshold": 0.012,
            "grasp_translation_threshold": 0.007,
            "rotation_threshold": 0.15,
            "alignment_steps": 4,
            "close_steps": 16,
            "hold_steps": 10,
            "lift_steps": 30,
            "lift_timeout_steps": 150,
            "lift_height": 0.04,
            "lift_tolerance": 0.003,
            "position_gain": 1.5,
            "align_position_gain": 4.0,
            "rotation_gain": 0.5,
            "max_hover_position_step": 0.02,
            "max_hover_rotation_step": 0.02,
            "align_fine_distance": 0.02,
            "max_align_position_step": 0.02,
            "max_align_fine_position_step": 0.008,
            "max_position_step": 0.004,
            "max_rotation_step": 0.01,
            "close_action": -0.43,
        }
        if controller_profile in ("j0_s2", "j1"):
            skill_kwargs.update({
                "intercept_lead_distance": 0.16,
                "intercept_align_lead_distance": 0.06,
                "pre_descent_height": 0.02,
                "align_target_xy_feedforward": True,
                "close_requires_raw_alignment": True,
                "target_xy_feedforward_gain": 1.0,
                "max_target_xy_feedforward_step": 0.002,
                "max_close_hold_xy_step": 0.008,
                "lift_height": 0.05,
                "lift_steps": 60,
                "lift_timeout_steps": 300,
                "lift_xy_gain": 0.01,
                "max_lift_xy_step": 0.002,
                "preserve_lift_relative_xy": True,
                "latch_lift_xy_feedforward": True,
                "lift_feedforward_x_only": True,
                "lift_feedforward_decay_steps": 20,
                "lift_geometry_relatch_height": 0.012,
                "settled_coupling_reference": True,
            })
        elif controller_profile == "static_pick":
            # A stationary supply object needs a safe high transit followed by
            # vertical descent.  Reusing the moving-belt progressive descent
            # drives the ZU5 diagonally into a poorly conditioned low posture.
            skill_kwargs.update({
                "intercept_x": None,
                "hover_height": 0.20,
                "pre_descent_height": None,
                "max_hover_position_step": 0.016,
                "max_align_position_step": 0.016,
                "lift_height": 0.05,
                "lift_steps": 60,
                "lift_timeout_steps": 300,
                "lift_xy_gain": 0.01,
                "max_lift_xy_step": 0.002,
                "preserve_lift_relative_xy": True,
                "settled_coupling_reference": True,
            })
        grasp_option = StableSphereResidualSkill(
            SphereGraspSkillConfig(**skill_kwargs),
            StableTargetConfig(
                translation_alpha=0.45,
                rotation_alpha=0.35,
                max_translation_jump=0.02,
                max_rotation_jump=0.35,
                stable_translation=0.004,
                stable_rotation=0.08,
                stable_steps=4,
                rotation_weight=0.05,
                switch_margin=0.01,
            ),
            translation_residual_limit=(
                0.05 if translation_residual_limit is None and self.pick_place
                else 0.10 if translation_residual_limit is None
                else float(translation_residual_limit)
            ),
            rotation_residual_limit=(
                0.025 if rotation_residual_limit is None and self.pick_place
                else 0.05 if rotation_residual_limit is None
                else float(rotation_residual_limit)
            ),
        )
        if self.pick_place:
            moving_bowl = self.task_variant == "static_sphere_moving_bowl"
            self.option = StableSpherePickPlaceSkill(
                grasp_option,
                SphereTransportPlaceConfig(
                    action_scale=ACTION_SCALE,
                    clear_height=0.20,
                    bowl_hover_height=0.16 if moving_bowl else 0.20,
                    release_height=0.11,
                    position_gain=4.0,
                    max_position_step=0.020 if moving_bowl else 0.016,
                    max_translation_action_delta=0.08 if moving_bowl else 0.04,
                    max_lower_xy_step=0.020 if moving_bowl else 0.002,
                    clear_tolerance=0.012,
                    hover_tolerance=0.025 if moving_bowl else 0.018,
                    release_tolerance=0.020 if moving_bowl else 0.012,
                    stable_steps=3,
                    release_steps=28,
                    close_action=-0.43,
                    open_action=1.0,
                ),
            )
        else:
            self.option = grasp_option
        self.last_action = np.zeros(7, dtype=np.float32)
        self.position_history = np.zeros((10, 11), dtype=np.float32)
        self.history_initialized = False
        self.ee_target_pos = JAKA_HOME_TCP.copy()
        self.ee_target_rot = Rotation.from_euler("x", np.pi)
        self.grasp_attached = False
        self.grasp_rel_pos = np.zeros(3)
        self.grasp_rel_rot = Rotation.identity()
        self.elapsed = 0.0
        self.initial_sphere_z = SPHERE_START_Z
        self.diagnostics = SphereDiagnostics()
        self._dual_contact_run = 0
        self._place_settle_count = 0
        self._placed_latched = False
        self._post_place_phase = 0
        self.reset(seed)

    def _tcp_pose(self) -> tuple[np.ndarray, Rotation]:
        return (
            self.data.site_xpos[self.tcp_site].copy(),
            Rotation.from_matrix(self.data.site_xmat[self.tcp_site].reshape(3, 3)),
        )

    def _select_sphere(self, index: int) -> None:
        if not 0 <= int(index) < self.num_spheres:
            raise IndexError(f"sphere index {index} outside [0, {self.num_spheres})")
        self.active_sphere_index = int(index)
        self.food_qpos = int(self.food_qpos_all[index])
        self.food_dof = int(self.food_dof_all[index])
        self.food_body = int(self.food_body_all[index])
        self.food_geom = int(self.food_geom_all[index])
        self.food_contype = int(self.food_contype_all[index])
        self.food_conaffinity = int(self.food_conaffinity_all[index])
        if self.queued_carriers:
            self.pallet_qpos = int(self.pallet_qpos_all[index])
            self.pallet_dof = int(self.pallet_dof_all[index])
            self.pallet_act = int(self.pallet_act_all[index])
        if self.queued_moving_bowls:
            self.moving_bowl_qpos = int(self.moving_bowl_qpos_all[index])
            self.moving_bowl_dof = int(self.moving_bowl_dof_all[index])
            self.destination_bowl_body = int(self.moving_bowl_body_all[index])

    def _set_pallet_targets(self, target_time: float) -> None:
        """Advance every carrier; queue mode uses uninterrupted world time."""
        if self.queued_carriers:
            spacing = (
                self.sphere_queue_spacing
                if self.queued_spheres
                else self.moving_bowl_queue_spacing
            )
            assert spacing is not None
            start = 0.0 if self.queued_spheres else -0.12
            for index, actuator in enumerate(self.pallet_act_all):
                target = start - spacing * index + self.belt_speed * target_time
                self.data.ctrl[actuator] = np.clip(
                    target,
                    self.model.actuator_ctrlrange[actuator, 0],
                    self.model.actuator_ctrlrange[actuator, 1],
                )
            return
        pallet_start = -0.12 if self.task_variant == "static_sphere_moving_bowl" else 0.0
        self.data.ctrl[self.pallet_act] = np.clip(
            pallet_start + self.belt_speed * target_time,
            self.model.actuator_ctrlrange[self.pallet_act, 0],
            self.model.actuator_ctrlrange[self.pallet_act, 1],
        )

    def _reset_single_call_state(self) -> None:
        """Reset policy/skill state without resetting robot, bowl, or old balls."""
        self.elapsed = 0.0
        self.last_action.fill(0.0)
        self.position_history.fill(0.0)
        self.history_initialized = False
        self.option.reset()
        self.diagnostics = SphereDiagnostics()
        self._dual_contact_run = 0
        self._place_settle_count = 0
        self._placed_latched = False
        self._post_place_phase = 0
        self.grasp_attached = False

    def prepare_sphere_call(
        self,
        index: int,
        *,
        sphere_start_x: float | None = None,
        sphere_start_y: float | None = None,
        belt_speed: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Spawn one pending ball and reset only the single-ball invocation.

        Previously placed balls remain untouched and physical.  The pallet is
        returned to the conveyor entry to model the next carrier arriving;
        neither the robot nor the destination bowl is reset.
        """
        if self.task_variant != "moving_sphere_static_bowl":
            raise RuntimeError("prepare_sphere_call is only valid for moving-sphere tasks")
        self._select_sphere(index)
        if belt_speed is not None:
            self.belt_speed = float(belt_speed)
        if self.queued_spheres:
            if sphere_start_x is not None or sphere_start_y is not None:
                raise ValueError("queued spheres are already in motion and cannot be respawned")
            self._reset_single_call_state()
            mujoco.mj_forward(self.model, self.data)
            self.ee_target_pos, self.ee_target_rot = self._tcp_pose()
            self.initial_sphere_z = float(self.data.xpos[self.food_body, 2])
            return self.observation(advance_history=True)
        start_x = self.sphere_start_x if sphere_start_x is None else float(sphere_start_x)
        start_y = self.sphere_start_y if sphere_start_y is None else float(sphere_start_y)
        self._reset_single_call_state()
        self.model.geom_contype[self.food_geom] = self.food_contype
        self.model.geom_conaffinity[self.food_geom] = self.food_conaffinity
        self.data.qpos[self.pallet_qpos] = 0.0
        self.data.qvel[self.pallet_dof] = self.belt_speed
        self.data.qpos[self.food_qpos : self.food_qpos + 7] = (
            start_x,
            start_y,
            SPHERE_START_Z,
            1.0,
            0.0,
            0.0,
            0.0,
        )
        self.data.qvel[self.food_dof : self.food_dof + 6] = (
            self.belt_speed,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        self.data.ctrl[self.finger_act] = AG95_OPEN
        self.data.ctrl[self.pallet_act] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.ee_target_pos, self.ee_target_rot = self._tcp_pose()
        self.initial_sphere_z = float(self.data.xpos[self.food_body, 2])
        return self.observation(advance_history=True)

    def prepare_static_sphere_moving_bowl_call(
        self,
        index: int,
    ) -> dict[str, np.ndarray]:
        """Select the next existing ball/bowl pair without resetting the world."""
        if not self.queued_moving_bowls:
            raise RuntimeError("moving-bowl queue mode is not enabled")
        self._select_sphere(index)
        self._reset_single_call_state()
        mujoco.mj_forward(self.model, self.data)
        self.ee_target_pos, self.ee_target_rot = self._tcp_pose()
        self.initial_sphere_z = float(self.data.xpos[self.food_body, 2])
        self.destination_bowl_pos = self.data.xpos[self.destination_bowl_body].copy()
        return self.observation(advance_history=True)

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._select_sphere(0)
        self._reset_single_call_state()
        for index, (qpos, dof, geom) in enumerate(
            zip(self.food_qpos_all, self.food_dof_all, self.food_geom_all)
        ):
            enabled = self.queued_spheres or self.queued_moving_bowls or index == 0
            self.model.geom_contype[geom] = self.food_contype_all[index] if enabled else 0
            self.model.geom_conaffinity[geom] = self.food_conaffinity_all[index] if enabled else 0
            if self.queued_spheres:
                assert self.sphere_queue_spacing is not None
                self.data.qpos[qpos : qpos + 7] = (
                    self.sphere_start_x - self.sphere_queue_spacing * index,
                    self.sphere_start_y,
                    SPHERE_START_Z,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                )
                self.data.qvel[dof : dof + 6] = (
                    self.belt_speed,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            elif self.queued_moving_bowls:
                column_order = (2, 1, 3, 0, 4)
                row = index // 5
                column = column_order[index % 5]
                self.data.qpos[qpos : qpos + 7] = (
                    0.34 + 0.055 * column,
                    -0.06 + 0.12 * row,
                    SPHERE_START_Z,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                )
                self.data.qvel[dof : dof + 6] = 0.0
            elif not enabled:
                self.data.qpos[qpos : qpos + 7] = (
                    -2.0 - 0.1 * index,
                    -2.0,
                    -1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                )
                self.data.qvel[dof : dof + 6] = 0.0
        self.data.qpos[self.arm_qpos] = JAKA_DEFAULT_Q
        self.data.qvel[self.arm_dof] = 0.0
        self.data.qpos[self.linkage_qpos] = AG95_OPEN
        self.data.qvel[self.linkage_dof] = 0.0
        pallet_start = -0.12 if self.task_variant == "static_sphere_moving_bowl" else 0.0
        if self.queued_carriers:
            spacing = (
                self.sphere_queue_spacing
                if self.queued_spheres
                else self.moving_bowl_queue_spacing
            )
            assert spacing is not None
            pallet_start = 0.0 if self.queued_spheres else -0.12
            for index, (qpos, dof) in enumerate(
                zip(self.pallet_qpos_all, self.pallet_dof_all)
            ):
                self.data.qpos[qpos] = pallet_start - spacing * index
                self.data.qvel[dof] = self.belt_speed
        else:
            self.data.qpos[self.pallet_qpos] = pallet_start
            self.data.qvel[self.pallet_dof] = self.belt_speed
        if self.task_variant == "static_sphere_moving_bowl":
            spacing = self.moving_bowl_queue_spacing or 0.0
            for index, (qpos, dof) in enumerate(
                zip(self.moving_bowl_qpos_all, self.moving_bowl_dof_all)
            ):
                self.data.qpos[qpos : qpos + 7] = (
                    0.08 - spacing * index,
                    0.30,
                    0.032,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                )
                self.data.qvel[dof : dof + 6] = (
                    self.belt_speed,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
        if not self.queued_spheres and not self.queued_moving_bowls:
            self.data.qpos[self.food_qpos : self.food_qpos + 7] = (
                self.sphere_start_x,
                self.sphere_start_y,
                SPHERE_START_Z,
                1.0,
                0.0,
                0.0,
                0.0,
            )
        if self.task_variant == "static_sphere_moving_bowl":
            self.data.qvel[self.food_dof : self.food_dof + 6] = 0.0
        elif not self.queued_spheres:
            self.data.qvel[self.food_dof : self.food_dof + 6] = (
                self.belt_speed,
                0,
                0,
                0,
                0,
                0,
            )
        self.data.ctrl[self.arm_act] = JAKA_DEFAULT_Q
        self.data.ctrl[self.finger_act] = AG95_OPEN
        self._set_pallet_targets(0.0)
        mujoco.mj_forward(self.model, self.data)
        self.ee_target_pos, self.ee_target_rot = self._tcp_pose()
        self.initial_sphere_z = float(self.data.xpos[self.food_body, 2])
        return self.observation(advance_history=True)

    def _guidance(self) -> tuple[dict[str, np.ndarray], torch.Tensor]:
        tcp_pos, tcp_rot = self._tcp_pose()
        sphere = torch.as_tensor(self.data.xpos[self.food_body], dtype=torch.float32).reshape(1, 3)
        tcp_position = torch.as_tensor(tcp_pos, dtype=torch.float32).reshape(1, 3)
        tcp_quat = torch.as_tensor(tcp_rot.as_quat(), dtype=torch.float32).reshape(1, 4)
        positions, rotations = sphere_grasp_frames(sphere, num_candidates=40, canonical_yaw=0.0)
        points = deterministic_grasp_points(positions, rotations, points_per_candidate=20, grasp_width=0.044)
        tcp_matrix = quat_xyzw_to_matrix(tcp_quat)
        points_ee = torch.einsum(
            "nij,ncpj->ncpi", tcp_matrix.transpose(-1, -2), points - tcp_position[:, None, None, :]
        )
        best_t, best_r, _ = nearest_grasp_delta(
            tcp_position, tcp_quat, positions, rotations, rotation_weight=0.05
        )
        all_t, all_r = grasp_pose_deltas(tcp_position, tcp_quat, positions, rotations)
        return {
            "points_ee": points_ee[0].numpy(),
            "best_pose_delta": torch.cat((best_t, best_r), dim=-1)[0].numpy(),
        }, torch.cat((all_t, all_r), dim=-1)

    def observation(self, *, advance_history: bool) -> dict[str, np.ndarray]:
        tcp_pos, tcp_rot = self._tcp_pose()
        guidance, _ = self._guidance()
        gripper = self.data.qpos[self.finger_qpos].copy()
        relative = self.data.xpos[self.food_body] - tcp_pos
        frame = np.concatenate((relative, gripper, self.last_action)).astype(np.float32)
        if advance_history:
            if not self.history_initialized:
                self.position_history[:] = frame
                self.history_initialized = True
            else:
                self.position_history[:-1] = self.position_history[1:].copy()
                self.position_history[-1] = frame
        return {
            "ee_pos": tcp_pos.astype(np.float32),
            "ee_quat": tcp_rot.as_quat().astype(np.float32),
            "gripper_pos": gripper.astype(np.float32),
            "joint_pos_rel": (self.data.qpos[self.arm_qpos] - JAKA_DEFAULT_Q).astype(np.float32),
            "last_action": self.last_action.copy(),
            **guidance,
            "position_action": self.position_history.copy(),
        }

    def _apply_ik(self) -> None:
        tcp_pos, tcp_rot = self._tcp_pose()
        error = np.concatenate(
            (self.ee_target_pos - tcp_pos, (self.ee_target_rot * tcp_rot.inv()).as_rotvec())
        )
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tcp_site)
        jacobian = np.vstack((jacp[:, self.arm_dof], jacr[:, self.arm_dof]))
        damping = 0.01
        dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(6), error)
        self.data.ctrl[self.arm_act] = np.clip(
            self.data.qpos[self.arm_qpos] + dq, JAKA_JOINT_LOW, JAKA_JOINT_HIGH
        )

    def _try_attach(self) -> None:
        if not self.grasp_assist or self.grasp_attached:
            return
        tcp_pos, tcp_rot = self._tcp_pose()
        master = float(self.data.qpos[self.finger_qpos][0])
        sphere_pos = self.data.xpos[self.food_body].copy()
        if np.linalg.norm(sphere_pos - tcp_pos) >= 0.03 or not 0.15 < master < 0.90:
            return
        sphere_rot = Rotation.from_quat(xyzw_from_wxyz(self.data.xquat[self.food_body]))
        self.grasp_rel_pos = tcp_rot.inv().apply(sphere_pos - tcp_pos)
        self.grasp_rel_rot = tcp_rot.inv() * sphere_rot
        self.grasp_attached = True
        self.model.geom_contype[self.food_geom] = 0
        self.model.geom_conaffinity[self.food_geom] = 0

    def _update_attached(self, dt: float) -> None:
        if not self.grasp_attached:
            return
        previous = self.data.qpos[self.food_qpos : self.food_qpos + 3].copy()
        tcp_pos, tcp_rot = self._tcp_pose()
        position = tcp_pos + tcp_rot.apply(self.grasp_rel_pos)
        rotation = tcp_rot * self.grasp_rel_rot
        quat = rotation.as_quat()
        self.data.qpos[self.food_qpos : self.food_qpos + 3] = position
        self.data.qpos[self.food_qpos + 3 : self.food_qpos + 7] = (quat[3], quat[0], quat[1], quat[2])
        self.data.qvel[self.food_dof : self.food_dof + 3] = (position - previous) / dt
        self.data.qvel[self.food_dof + 3 : self.food_dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _update_contact_diagnostics(self, time_s: float) -> None:
        left = right = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 == self.food_geom:
                other = geom2
            elif geom2 == self.food_geom:
                other = geom1
            else:
                continue
            body = int(self.model.geom_bodyid[other])
            left |= body == self.left_finger_body
            right |= body == self.right_finger_body
        d = self.diagnostics
        d.ever_left_contact |= left
        d.ever_right_contact |= right
        d.current_left_contact = left
        d.current_right_contact = right
        d.current_dual_contact = left and right
        d.min_finger_gap = min(d.min_finger_gap, float(self.data.qpos[self.finger_qpos][0]))
        if left and right:
            d.ever_dual_contact = True
            d.dual_contact_substeps += 1
            self._dual_contact_run += 1
            d.max_dual_contact_run = max(d.max_dual_contact_run, self._dual_contact_run)
            if d.first_dual_contact_time is None:
                d.first_dual_contact_time = float(time_s)
        else:
            self._dual_contact_run = 0

    def return_to_intercept_ready(
        self,
        *,
        retreat_height: float = 0.20,
        tolerance: float = 0.015,
        max_seconds: float = 4.0,
        on_step: Callable[[], None] | None = None,
    ) -> dict[str, float | bool]:
        """Retreat and return Home between independent policy invocations.

        This advances the same MuJoCo world, so all balls already placed in
        the bowl continue settling and colliding.  It resets no scene state.
        """
        start_time = float(self.data.time)
        waypoints = []
        tcp_pos, _ = self._tcp_pose()
        waypoints.append(np.asarray((tcp_pos[0], tcp_pos[1], retreat_height), dtype=float))
        waypoints.append(JAKA_HOME_TCP.copy())
        waypoint_index = 0
        while float(self.data.time) - start_time < float(max_seconds):
            tcp_pos, tcp_rot = self._tcp_pose()
            target = waypoints[waypoint_index]
            error = target - tcp_pos
            if np.linalg.norm(error) <= tolerance:
                waypoint_index += 1
                if waypoint_index >= len(waypoints):
                    break
                continue
            position_step = np.clip(4.0 * error, -0.016, 0.016)
            self.ee_target_pos = tcp_pos + position_step
            self.ee_target_rot = tcp_rot
            for _ in range(self.physics_steps):
                self._apply_ik()
                self.data.ctrl[self.finger_act] = AG95_OPEN
                self._set_pallet_targets(
                    float(self.data.time) + self.model.opt.timestep
                    if self.queued_carriers
                    else self.elapsed + self.model.opt.timestep
                )
                arm_before = self.data.qpos[self.arm_qpos].copy()
                mujoco.mj_step(self.model, self.data)
                arm_delta = np.clip(
                    self.data.qpos[self.arm_qpos] - arm_before,
                    -JAKA_JOINT_VELOCITY * self.model.opt.timestep,
                    JAKA_JOINT_VELOCITY * self.model.opt.timestep,
                )
                self.data.qpos[self.arm_qpos] = np.clip(
                    arm_before + arm_delta, JAKA_JOINT_LOW, JAKA_JOINT_HIGH
                )
                self.data.qvel[self.arm_dof] = arm_delta / self.model.opt.timestep
                mujoco.mj_forward(self.model, self.data)
            if on_step is not None:
                on_step()
        final_distance = float(np.linalg.norm(self._tcp_pose()[0] - JAKA_HOME_TCP))
        return {
            "success": final_distance <= tolerance,
            "seconds": float(self.data.time) - start_time,
            "home_distance": final_distance,
        }

    def step(self, policy_action: np.ndarray) -> StepResult:
        policy_action = np.asarray(policy_action, dtype=np.float32).reshape(7)
        _, candidates = self._guidance()
        tcp_pos, _ = self._tcp_pose()
        policy_tensor = torch.from_numpy(policy_action).reshape(1, 7)
        tcp_tensor = torch.as_tensor(tcp_pos, dtype=torch.float32).reshape(1, 3)
        finger_tensor = torch.as_tensor(
            self.data.qpos[self.finger_qpos], dtype=torch.float32
        ).reshape(1, 1)
        if self.pick_place:
            if self.task_variant == "static_sphere_moving_bowl":
                self.destination_bowl_pos = self.data.xpos[self.destination_bowl_body].copy()
            bowl_tensor = torch.as_tensor(self.destination_bowl_pos, dtype=torch.float32).reshape(1, 3)
            filtered, grasp_phase, phase = self.option(
                policy_tensor, candidates, tcp_tensor, finger_tensor, bowl_tensor
            )
            phase_name = StableSpherePickPlaceSkill.PHASE_NAMES[int(phase[0])]
            grasp_phase_name = StableSphereGraspSkill.PHASE_NAMES[int(grasp_phase[0])]
            triggered = int(grasp_phase[0]) > StableSphereGraspSkill.HOVER
        else:
            filtered, phase, _, _ = self.option(
                policy_tensor, candidates, tcp_tensor, finger_tensor
            )
            grasp_phase = phase
            phase_name = StableSphereGraspSkill.PHASE_NAMES[int(phase[0])]
            grasp_phase_name = phase_name
            triggered = int(phase[0]) > StableSphereGraspSkill.HOVER
        action = filtered[0].numpy().clip(-1.0, 1.0)
        if (
            self.task_variant == "static_sphere_moving_bowl"
            and self.pick_place
            and int(phase[0]) == StableSpherePickPlaceSkill.DONE
        ):
            # Complete the original task semantics after release: retreat
            # vertically clear of the moving bowl, then return to the JAKA
            # Home TCP.  The ball/bowl continue under MuJoCo contact dynamics.
            tcp_now, _ = self._tcp_pose()
            if self._post_place_phase == 0:
                self._post_place_phase = 1
            if self._post_place_phase == 1:
                target = np.asarray((tcp_now[0], tcp_now[1], 0.20), dtype=float)
                phase_name = "retreat"
                if tcp_now[2] >= 0.185:
                    self._post_place_phase = 2
            else:
                target = JAKA_HOME_TCP
                phase_name = "home" if self._post_place_phase == 2 else "done"
                if np.linalg.norm(tcp_now - JAKA_HOME_TCP) <= 0.025:
                    self._post_place_phase = 3
                    phase_name = "done"
            step_to_target = np.clip(3.0 * (target - tcp_now), -0.016, 0.016)
            action[:] = 0.0
            action[:3] = np.clip(step_to_target / ACTION_SCALE, -1.0, 1.0)
            action[-1] = 1.0
        processed = ACTION_SCALE * action[:6]
        tcp_pos, tcp_rot = self._tcp_pose()
        self.ee_target_pos = tcp_pos + processed[:3]
        self.ee_target_rot = Rotation.from_rotvec(processed[3:]) * tcp_rot
        finger_target = 0.5 * AG95_CLOSED * (1.0 - float(action[6]))

        for substep in range(self.physics_steps):
            self._apply_ik()
            self.data.ctrl[self.finger_act] = finger_target
            target_t = self.elapsed + (substep + 1) * self.model.opt.timestep
            pallet_target_time = (
                float(self.data.time) + self.model.opt.timestep
                if self.queued_carriers
                else target_t
            )
            self._set_pallet_targets(pallet_target_time)
            arm_before = self.data.qpos[self.arm_qpos].copy()
            mujoco.mj_step(self.model, self.data)
            arm_delta = np.clip(
                self.data.qpos[self.arm_qpos] - arm_before,
                -JAKA_JOINT_VELOCITY * self.model.opt.timestep,
                JAKA_JOINT_VELOCITY * self.model.opt.timestep,
            )
            self.data.qpos[self.arm_qpos] = np.clip(
                arm_before + arm_delta, JAKA_JOINT_LOW, JAKA_JOINT_HIGH
            )
            self.data.qvel[self.arm_dof] = arm_delta / self.model.opt.timestep
            mujoco.mj_forward(self.model, self.data)
            self._update_contact_diagnostics(target_t)
            if finger_target > 0.05:
                self._try_attach()
                self._update_attached(self.model.opt.timestep)

        self.elapsed += POLICY_DT
        self.last_action = action.astype(np.float32)
        obs = self.observation(advance_history=True)
        translation = float(np.linalg.norm(obs["best_pose_delta"][:3]))
        rotation = float(np.linalg.norm(obs["best_pose_delta"][3:]))
        lift = float(self.data.xpos[self.food_body, 2] - self.initial_sphere_z)
        self.diagnostics.min_translation = min(self.diagnostics.min_translation, translation)
        self.diagnostics.min_rotation = min(self.diagnostics.min_rotation, rotation)
        self.diagnostics.max_lift = max(self.diagnostics.max_lift, lift)
        self.diagnostics.option_triggered |= triggered
        physical_grasp_secured = self.grasp_attached or (
            not self.grasp_assist
            and self.diagnostics.current_dual_contact
            and self.diagnostics.max_dual_contact_run
            >= max(1, round(0.05 / self.model.opt.timestep))
        )
        sphere_pos = self.data.xpos[self.food_body].copy()
        bowl_delta = sphere_pos - self.destination_bowl_pos
        sphere_speed = float(np.linalg.norm(self.data.qvel[self.food_dof : self.food_dof + 3]))
        bowl_linear_velocity = self.data.cvel[self.destination_bowl_body, 3:].copy()
        relative_sphere_speed = float(
            np.linalg.norm(
                self.data.qvel[self.food_dof : self.food_dof + 3] - bowl_linear_velocity
            )
        )
        if self.pick_place:
            released = bool(float(self.data.qpos[self.finger_qpos][0]) < 0.10)
            in_bowl = bool(
                np.linalg.norm(bowl_delta[:2]) < 0.06
                and 0.005 < bowl_delta[2] < 0.08
                and released
            )
            # A moving-bowl placement is settled in the bowl frame, not the
            # world frame.  Requiring absolute speed below 5 cm/s would make
            # a correct 0.10 m/s transport impossible by definition.
            settle_speed = (
                relative_sphere_speed
                if self.task_variant == "static_sphere_moving_bowl"
                else sphere_speed
            )
            settled = in_bowl and settle_speed < 0.05
            self._place_settle_count = self._place_settle_count + 1 if settled else 0
            settled_in_bowl = self._place_settle_count >= 5
            if self.task_variant == "static_sphere_moving_bowl":
                self._placed_latched |= bool(settled_in_bowl)
                home_distance = float(np.linalg.norm(self._tcp_pose()[0] - JAKA_HOME_TCP))
                if (
                    self._placed_latched
                    and self._post_place_phase == 3
                    and home_distance <= 0.025
                ):
                    outcome = "strict_place_home_success"
                elif sphere_pos[2] < -0.05:
                    outcome = "sphere_off_table"
                elif self.elapsed >= self.max_seconds:
                    outcome = "time_out"
                else:
                    outcome = None
            else:
                home_distance = float(np.linalg.norm(self._tcp_pose()[0] - JAKA_HOME_TCP))
                outcome = moving_sphere_place_outcome(
                    settled_in_bowl=settled_in_bowl,
                    grasp_ever_secured=self.diagnostics.ever_dual_contact or self.grasp_attached,
                    sphere_x=float(sphere_pos[0]),
                    sphere_z=float(sphere_pos[2]),
                    elapsed=self.elapsed,
                    max_seconds=self.max_seconds,
                )
        else:
            in_bowl = False
            settled_in_bowl = False
            home_distance = float(np.linalg.norm(self._tcp_pose()[0] - JAKA_HOME_TCP))
            outcome = moving_sphere_outcome(
                grasp_attached=physical_grasp_secured,
                lift=lift,
                sphere_x=float(sphere_pos[0]),
                sphere_z=float(sphere_pos[2]),
                elapsed=self.elapsed,
                max_seconds=self.max_seconds,
            )
        info: dict[str, float | bool | str | None] = {
            "time": round(self.elapsed, 4),
            "sphere_x": float(sphere_pos[0]),
            "sphere_z": float(sphere_pos[2]),
            "tcp_distance": translation,
            "rotation_error": rotation,
            "option_phase": phase_name,
            "grasp_phase": grasp_phase_name,
            "pick_place": self.pick_place,
            "grasp_assist_attached": self.grasp_attached,
            "physical_grasp_secured": physical_grasp_secured,
            "max_lift": self.diagnostics.max_lift,
            "ever_left_contact": self.diagnostics.ever_left_contact,
            "ever_right_contact": self.diagnostics.ever_right_contact,
            "ever_dual_contact": self.diagnostics.ever_dual_contact,
            "current_dual_contact": self.diagnostics.current_dual_contact,
            "first_dual_contact_time": self.diagnostics.first_dual_contact_time,
            "max_dual_contact_seconds": self.diagnostics.max_dual_contact_run * self.model.opt.timestep,
            "ag95_master": float(self.data.qpos[self.finger_qpos][0]),
            "sphere_speed": sphere_speed,
            "sphere_to_bowl_speed": relative_sphere_speed,
            "sphere_to_bowl_xy": float(np.linalg.norm(bowl_delta[:2])),
            "in_bowl": in_bowl,
            "settled_in_bowl": settled_in_bowl,
            "place_settle_count": self._place_settle_count,
            "placed_latched": self._placed_latched,
            "home_distance": home_distance,
        }
        return StepResult(obs, outcome is not None, outcome, info)

    def close(self) -> None:
        pass


DEFAULT_CHECKPOINT = ROOT / (
    "artifacts/jaka_zu5_ag95_sphere/teachers/"
    "jaka_zu5_ag95_sphere_j0_s1_probe500k/checkpoints/ppo_teacher_final.pt"
)
