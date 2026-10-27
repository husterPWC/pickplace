"""Closed-loop MuJoCo contract for the speed-free moving-sphere teacher.

This module is deliberately separate from :mod:`mujoco_sim.env`: the latter
is the accepted historical fixed-food/moving-bowl v3c task.  Here the sphere
moves on the conveyor, the destination bowl is static, and the actor receives
exactly the deployable G0-S0 observation groups (no explicit velocity or
contact signal).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from mujoco_sim.env import (
    DEFAULT_Q,
    HOME_TCP,
    JOINT_HIGH,
    JOINT_LOW,
    JOINT_VELOCITY,
    POLICY_DT,
    ROOT,
    SCENE,
    StepResult,
    xyzw_from_wxyz,
)
from pickplace.grasp_geometry import (
    deterministic_grasp_points,
    grasp_pose_deltas,
    nearest_grasp_delta,
    quat_xyzw_to_matrix,
    sphere_grasp_frames,
)
from pickplace.grasp_skills import (
    SphereCloseHoldLiftOptionConfig,
    SphereGraspSkillConfig,
    SphereTransportPlaceConfig,
    StableSphereCloseHoldLiftOption,
    StableSphereGraspSkill,
    StableSpherePickPlaceSkill,
    StableSphereResidualSkill,
    StableTargetConfig,
)


ACTION_SCALE = 0.02
BELT_SPEED = 0.01
BELT_Y = 0.30
BELT_ENTRY_X = 0.20
ZONE_END_X = 0.65
SPHERE_RADIUS = 0.02
SPHERE_START_Z = 0.05
MAX_SECONDS = 45.0
# The Isaac skill requests a 4 cm TCP lift.  MuJoCo's joint-space tracking
# leaves the attached sphere about 0.6 mm below the unchanged 3 cm acceptance
# threshold, so the bridge adds 5 mm of execution margin.  This does not relax
# the measured sphere-lift success criterion or alter the trained policy.
MUJOCO_RESIDUAL_LIFT_HEIGHT = 0.045


def moving_sphere_outcome(
    *,
    grasp_attached: bool,
    lift: float,
    sphere_x: float,
    sphere_z: float,
    elapsed: float,
    max_seconds: float = MAX_SECONDS,
) -> str | None:
    """Mirror Isaac's lift-stage deadline contract.

    Crossing the downstream line is a failure only before a stable grasp.  An
    attached sphere must be allowed to finish HOLD/LIFT beyond that line.
    """
    if grasp_attached and lift >= 0.03:
        return "strict_lift_success"
    if not grasp_attached and sphere_x > ZONE_END_X:
        return "sphere_exited_zone"
    if elapsed >= max_seconds:
        return "time_out"
    if sphere_z < -0.03:
        return "sphere_off_table"
    return None


def moving_sphere_place_outcome(
    *,
    settled_in_bowl: bool,
    grasp_ever_secured: bool,
    sphere_x: float,
    sphere_z: float,
    elapsed: float,
    max_seconds: float = MAX_SECONDS,
) -> str | None:
    """Full-task termination without exposing privileged state to the actor."""
    if settled_in_bowl:
        return "strict_place_success"
    if not grasp_ever_secured and sphere_x > ZONE_END_X:
        return "sphere_exited_zone"
    if sphere_z < -0.03:
        return "sphere_off_table"
    if elapsed >= max_seconds:
        return "time_out"
    return None


class GraspPointGRUActor(nn.Module):
    """Deterministic reconstruction of the G0-S0 grasp-point PPO actor."""

    def __init__(self, checkpoint: Path):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        network = payload["config"]["network"]
        expected_keys = [
            ["proprio", "ee_pos"],
            ["proprio", "ee_quat"],
            ["proprio", "gripper_pos"],
            ["proprio", "joint_pos_rel"],
            ["proprio", "last_action"],
            ["grasp_guidance", "points_ee"],
            ["grasp_guidance", "best_pose_delta"],
            ["position_history", "position_action"],
        ]
        if network.get("architecture") != "grasp_point_gru" or network.get("actor_in_keys") != expected_keys:
            raise ValueError("checkpoint is not the moving-sphere grasp_point_gru actor")
        if list(network["hidden"]) != [512, 256, 128]:
            raise ValueError(f"unexpected hidden layers: {network['hidden']}")

        source = payload["actor"]
        prefix = "module.0.module.body."
        body_state = {name.removeprefix(prefix): value for name, value in source.items() if name.startswith(prefix)}
        point_hidden = int(network.get("point_hidden", 64))
        temporal_hidden = int(network.get("temporal_hidden", 64))
        temporal_input = int(body_state["gru.weight_ih_l0"].shape[1])
        fusion_input = int(body_state["fusion.0.weight"].shape[1])
        self.point_encoder = nn.Sequential(
            nn.Linear(3, point_hidden), nn.ELU(), nn.Linear(point_hidden, point_hidden), nn.ELU()
        )
        self.grasp_encoder = nn.Sequential(nn.Linear(point_hidden, point_hidden), nn.ELU())
        self.gru = nn.GRU(temporal_input, temporal_hidden, batch_first=True)
        # Franka and JAKA have different robot/gripper state widths.  Rebuild
        # the fusion layer from the checkpoint itself instead of assuming the
        # historical Franka 31-vector contract.
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 7),
        )

        missing, unexpected = self.load_state_dict(body_state, strict=False)
        if missing or unexpected:
            raise ValueError(f"actor state mismatch: missing={missing}, unexpected={unexpected}")
        self.eval()

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        points = observation["points_ee"].float()
        point_features = self.point_encoder(points).max(dim=-2).values
        grasp_features = self.grasp_encoder(point_features).max(dim=-2).values
        temporal_features, _ = self.gru(observation["position_action"].float())
        vectors = torch.cat(
            [
                observation["ee_pos"],
                observation["ee_quat"],
                observation["gripper_pos"],
                observation["joint_pos_rel"],
                observation["last_action"],
                observation["best_pose_delta"],
                grasp_features,
                temporal_features[:, -1],
            ],
            dim=-1,
        )
        return torch.tanh(self.fusion(vectors))

    @torch.no_grad()
    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        batch = {key: torch.from_numpy(value).float().unsqueeze(0) for key, value in observation.items()}
        return self(batch)[0].cpu().numpy()


@dataclass
class SphereDiagnostics:
    min_translation: float = float("inf")
    min_rotation: float = float("inf")
    max_lift: float = 0.0
    option_triggered: bool = False
    ever_left_contact: bool = False
    ever_right_contact: bool = False
    ever_dual_contact: bool = False
    current_left_contact: bool = False
    current_right_contact: bool = False
    current_dual_contact: bool = False
    first_dual_contact_time: float | None = None
    dual_contact_substeps: int = 0
    max_dual_contact_run: int = 0
    min_finger_gap: float = float("inf")


class FrankaMovingSphereMuJoCo:
    """One true closed-loop moving-sphere MuJoCo environment."""

    def __init__(
        self,
        seed: int = 0,
        belt_speed: float = BELT_SPEED,
        grasp_assist: bool = True,
        controller: str = "option",
        intercept_lead_distance: float = 0.08,
        intercept_align_lead_distance: float = 0.04,
        pre_descent_height: float | None = None,
        lift_steps: int = 30,
        lift_height: float = MUJOCO_RESIDUAL_LIFT_HEIGHT,
        lift_xy_gain: float | None = None,
        max_lift_xy_step: float | None = None,
        preserve_lift_relative_xy: bool = False,
        target_xy_feedforward_gain: float = 0.0,
        max_target_xy_feedforward_step: float | None = None,
        align_target_xy_feedforward: bool = False,
        close_requires_raw_alignment: bool = False,
        max_close_hold_xy_step: float | None = None,
        latch_lift_xy_feedforward: bool = False,
        lift_geometry_relatch_height: float | None = None,
        sphere_start_x: float = BELT_ENTRY_X,
        sphere_start_y: float = BELT_Y,
        belt_upstream_x: float | None = None,
        pallet_start_x: float | None = None,
        lift_timeout_steps: int = 300,
        max_seconds: float = MAX_SECONDS,
        pick_place: bool = False,
        destination_bowl_pos: tuple[float, float, float] = (0.45, 0.0, 0.0),
        translation_residual_limit: float = 0.10,
        rotation_residual_limit: float = 0.05,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.belt_speed = float(belt_speed)
        self.sphere_start_x = float(sphere_start_x)
        self.sphere_start_y = float(sphere_start_y)
        self.max_seconds = float(max_seconds)
        if self.max_seconds <= 0.0:
            raise ValueError("max_seconds must be positive")
        self.grasp_assist = bool(grasp_assist)
        self.pick_place = bool(pick_place)
        self.destination_bowl_pos = np.asarray(destination_bowl_pos, dtype=float)
        if self.destination_bowl_pos.shape != (3,):
            raise ValueError("destination_bowl_pos must contain three values")
        self.controller_lift_height = float(lift_height)
        belt_visual = self.model.geom("belt_visual").id
        belt_downstream_x = float(
            self.model.geom_pos[belt_visual, 0] + self.model.geom_size[belt_visual, 0]
        )
        if belt_upstream_x is None:
            self.belt_upstream_x = float(
                self.model.geom_pos[belt_visual, 0] - self.model.geom_size[belt_visual, 0]
            )
        else:
            self.belt_upstream_x = float(belt_upstream_x)
            if self.belt_upstream_x >= belt_downstream_x:
                raise ValueError("belt_upstream_x must be below the existing downstream edge")
            self.model.geom_pos[belt_visual, 0] = 0.5 * (
                self.belt_upstream_x + belt_downstream_x
            )
            self.model.geom_size[belt_visual, 0] = 0.5 * (
                belt_downstream_x - self.belt_upstream_x
            )
        if controller not in {"option", "residual_baseline"}:
            raise ValueError("controller must be 'option' or 'residual_baseline'")
        self.controller_mode = controller
        self.physics_steps = round(POLICY_DT / self.model.opt.timestep)

        self.arm_qpos = np.asarray([self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_dof = np.asarray([self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.finger_qpos = np.asarray([self.model.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)])
        self.finger_dof = np.asarray([self.model.joint(f"finger_joint{i}").dofadr[0] for i in (1, 2)])
        self.robot_qpos = np.concatenate([self.arm_qpos, self.finger_qpos])
        self.pallet_qpos = int(self.model.joint("pallet_slide").qposadr[0])
        self.pallet_dof = int(self.model.joint("pallet_slide").dofadr[0])
        self.pallet_joint = self.model.joint("pallet_slide").id
        self.bowl_qpos = int(self.model.joint("bowl_free").qposadr[0])
        self.bowl_dof = int(self.model.joint("bowl_free").dofadr[0])
        self.food_qpos = int(self.model.joint("food_free").qposadr[0])
        self.food_dof = int(self.model.joint("food_free").dofadr[0])
        self.arm_act = np.asarray([self.model.actuator(f"arm{i}").id for i in range(1, 8)])
        self.finger_act = np.asarray([self.model.actuator(f"finger{i}").id for i in (1, 2)])
        self.pallet_act = self.model.actuator("pallet_drive").id
        pallet_body = self.model.body("pallet").id
        pallet_base_x = float(self.model.body_pos[pallet_body, 0])
        resolved_pallet_start_x = pallet_base_x if pallet_start_x is None else float(pallet_start_x)
        self.pallet_start_x = resolved_pallet_start_x
        self.pallet_initial_qpos = resolved_pallet_start_x - pallet_base_x
        # The historical joint/actuator lower bound only covered the original
        # x=0.20 m spawn.  Expand it locally for this model instance so the
        # long-belt variant can start the carrier beneath the sphere.
        lower_margin = self.pallet_initial_qpos - 0.02
        self.model.jnt_range[self.pallet_joint, 0] = min(
            self.model.jnt_range[self.pallet_joint, 0], lower_margin
        )
        self.model.actuator_ctrlrange[self.pallet_act, 0] = min(
            self.model.actuator_ctrlrange[self.pallet_act, 0], lower_margin
        )
        self.tcp_site = self.model.site("tcp").id
        self.food_body = self.model.body("food").id
        self.food_geom = self.model.geom("food_geom").id
        self.destination_bowl_body = self.model.body("supply_bowl").id
        self.model.body_pos[self.destination_bowl_body] = self.destination_bowl_pos
        self.left_finger_body = self.model.body("left_finger").id
        self.right_finger_body = self.model.body("right_finger").id
        self.food_contype = int(self.model.geom_contype[self.food_geom])
        self.food_conaffinity = int(self.model.geom_conaffinity[self.food_geom])

        if controller == "residual_baseline":
            grasp_controller = StableSphereResidualSkill(
                SphereGraspSkillConfig(
                    action_scale=ACTION_SCALE,
                    intercept_x=0.45,
                    intercept_lead_distance=intercept_lead_distance,
                    intercept_align_lead_distance=intercept_align_lead_distance,
                    hover_height=0.08,
                    pre_descent_height=pre_descent_height,
                    hover_translation_threshold=0.012,
                    grasp_translation_threshold=0.007,
                    rotation_threshold=0.15,
                    alignment_steps=4,
                    close_steps=16,
                    hold_steps=10,
                    lift_steps=lift_steps,
                    # MuJoCo's joint-space IK has a small steady-state tracking
                    # offset relative to Isaac's differential IK.  Keep the
                    # strict 3 cm sphere-lift acceptance unchanged and give the
                    # execution target a 5 mm cross-engine margin.
                    lift_timeout_steps=lift_timeout_steps,
                    lift_height=lift_height,
                    lift_tolerance=0.003,
                    lift_xy_gain=lift_xy_gain,
                    max_lift_xy_step=max_lift_xy_step,
                    preserve_lift_relative_xy=preserve_lift_relative_xy,
                    target_xy_feedforward_gain=target_xy_feedforward_gain,
                    max_target_xy_feedforward_step=max_target_xy_feedforward_step,
                    align_target_xy_feedforward=align_target_xy_feedforward,
                    close_requires_raw_alignment=close_requires_raw_alignment,
                    max_close_hold_xy_step=max_close_hold_xy_step,
                    latch_lift_xy_feedforward=latch_lift_xy_feedforward,
                    lift_geometry_relatch_height=lift_geometry_relatch_height,
                    position_gain=1.5,
                    align_position_gain=4.0,
                    rotation_gain=0.5,
                    max_hover_position_step=0.02,
                    max_hover_rotation_step=0.02,
                    align_fine_distance=0.02,
                    max_align_position_step=0.02,
                    max_align_fine_position_step=0.008,
                    max_position_step=0.004,
                    max_rotation_step=0.01,
                    close_action=-0.15,
                ),
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
                translation_residual_limit=translation_residual_limit,
                rotation_residual_limit=rotation_residual_limit,
            )
            if self.pick_place:
                self.option = StableSpherePickPlaceSkill(
                    grasp_controller,
                    SphereTransportPlaceConfig(
                        action_scale=ACTION_SCALE,
                        clear_height=0.20,
                        bowl_hover_height=0.20,
                        release_height=0.11,
                        position_gain=4.0,
                        max_position_step=0.016,
                        max_translation_action_delta=0.04,
                        clear_tolerance=0.012,
                        hover_tolerance=0.018,
                        release_tolerance=0.012,
                        stable_steps=3,
                        release_steps=28,
                        close_action=-0.15,
                        open_action=1.0,
                    ),
                )
            else:
                self.option = grasp_controller
        else:
            if self.pick_place:
                raise ValueError("pick_place requires controller='residual_baseline'")
            self.option = StableSphereCloseHoldLiftOption(SphereCloseHoldLiftOptionConfig())
        self.last_action = np.zeros(7, dtype=np.float32)
        self.position_history = np.zeros((10, 12), dtype=np.float32)
        self.history_initialized = False
        self.ee_target_pos = HOME_TCP.copy()
        self.ee_target_rot = Rotation.identity()
        self.grasp_attached = False
        self.grasp_rel_pos = np.zeros(3)
        self.grasp_rel_rot = Rotation.identity()
        self.elapsed = 0.0
        self.initial_sphere_z = SPHERE_START_Z
        self.diagnostics = SphereDiagnostics()
        self._dual_contact_run = 0
        self._place_settle_count = 0
        self.reset(seed)

    def _tcp_pose(self) -> tuple[np.ndarray, Rotation]:
        return (
            self.data.site_xpos[self.tcp_site].copy(),
            Rotation.from_matrix(self.data.site_xmat[self.tcp_site].reshape(3, 3)),
        )

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.elapsed = 0.0
        self.last_action.fill(0.0)
        self.position_history.fill(0.0)
        self.history_initialized = False
        self.option.reset()
        self.diagnostics = SphereDiagnostics()
        self._dual_contact_run = 0
        self._place_settle_count = 0
        self.grasp_attached = False
        self.model.geom_contype[self.food_geom] = self.food_contype
        self.model.geom_conaffinity[self.food_geom] = self.food_conaffinity

        self.data.qpos[self.arm_qpos] = DEFAULT_Q[:7]
        self.data.qvel[self.arm_dof] = 0.0
        self.data.qpos[self.finger_qpos] = 0.04
        self.data.qvel[self.finger_dof] = 0.0
        self.data.qpos[self.pallet_qpos] = self.pallet_initial_qpos
        self.data.qvel[self.pallet_dof] = self.belt_speed
        # The scene's shallow fixed supply tray at (0.45, -0.10) is the
        # destination bowl. Move the old task's free bowl out of view.
        self.data.qpos[self.bowl_qpos : self.bowl_qpos + 7] = (2.0, 2.0, 0.032, 1, 0, 0, 0)
        self.data.qvel[self.bowl_dof : self.bowl_dof + 6] = 0.0
        self.data.qpos[self.food_qpos : self.food_qpos + 7] = (
            self.sphere_start_x,
            self.sphere_start_y,
            SPHERE_START_Z,
            1,
            0,
            0,
            0,
        )
        self.data.qvel[self.food_dof : self.food_dof + 6] = (self.belt_speed, 0, 0, 0, 0, 0)
        self.data.ctrl[self.arm_act] = self.data.qpos[self.arm_qpos]
        self.data.ctrl[self.finger_act] = 0.04
        self.data.ctrl[self.pallet_act] = self.pallet_initial_qpos
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
        best = torch.cat((best_t, best_r), dim=-1)
        candidates = torch.cat((all_t, all_r), dim=-1)
        return {
            "points_ee": points_ee[0].numpy(),
            "best_pose_delta": best[0].numpy(),
        }, candidates

    def observation(self, *, advance_history: bool) -> dict[str, np.ndarray]:
        tcp_pos, tcp_rot = self._tcp_pose()
        guidance, _ = self._guidance()
        relative = self.data.xpos[self.food_body] - tcp_pos
        frame = np.concatenate((relative, self.data.qpos[self.finger_qpos], self.last_action)).astype(np.float32)
        if advance_history:
            if not self.history_initialized:
                self.position_history[:] = frame
                self.history_initialized = True
            else:
                self.position_history[:-1] = self.position_history[1:].copy()
                self.position_history[-1] = frame
        obs = {
            "ee_pos": tcp_pos.astype(np.float32),
            "ee_quat": tcp_rot.as_quat().astype(np.float32),
            "gripper_pos": self.data.qpos[self.finger_qpos].astype(np.float32),
            "joint_pos_rel": (self.data.qpos[self.robot_qpos] - DEFAULT_Q).astype(np.float32),
            "last_action": self.last_action.copy(),
            **guidance,
            "position_action": self.position_history.copy(),
        }
        return obs

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
        self.data.ctrl[self.arm_act] = np.clip(self.data.qpos[self.arm_qpos] + dq, JOINT_LOW, JOINT_HIGH)

    def _try_attach(self) -> None:
        if not self.grasp_assist or self.grasp_attached:
            return
        tcp_pos, tcp_rot = self._tcp_pose()
        fingers = self.data.qpos[self.finger_qpos]
        sphere_pos = self.data.xpos[self.food_body].copy()
        if np.linalg.norm(sphere_pos - tcp_pos) >= 0.03 or not np.all((fingers > 0.003) & (fingers < 0.03)):
            return
        sphere_rot = Rotation.from_quat(xyzw_from_wxyz(self.data.xquat[self.food_body]))
        self.grasp_rel_pos = tcp_rot.inv().apply(sphere_pos - tcp_pos)
        self.grasp_rel_rot = tcp_rot.inv() * sphere_rot
        self.grasp_attached = True
        self.model.geom_contype[self.food_geom] = 0
        self.model.geom_conaffinity[self.food_geom] = 0

    def _update_contact_diagnostics(self, time_s: float) -> None:
        """Record physical sphere/finger contacts without exposing them to the actor."""
        left = False
        right = False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self.food_geom:
                other = geom2
            elif geom2 == self.food_geom:
                other = geom1
            else:
                continue
            body = int(self.model.geom_bodyid[other])
            left |= body == self.left_finger_body
            right |= body == self.right_finger_body

        diagnostics = self.diagnostics
        diagnostics.ever_left_contact |= left
        diagnostics.ever_right_contact |= right
        diagnostics.current_left_contact = left
        diagnostics.current_right_contact = right
        diagnostics.current_dual_contact = left and right
        diagnostics.min_finger_gap = min(
            diagnostics.min_finger_gap,
            float(self.data.qpos[self.finger_qpos].sum()),
        )
        if left and right:
            diagnostics.ever_dual_contact = True
            diagnostics.dual_contact_substeps += 1
            self._dual_contact_run += 1
            diagnostics.max_dual_contact_run = max(
                diagnostics.max_dual_contact_run, self._dual_contact_run
            )
            if diagnostics.first_dual_contact_time is None:
                diagnostics.first_dual_contact_time = float(time_s)
        else:
            self._dual_contact_run = 0

    def _update_attached(self, dt: float) -> None:
        if not self.grasp_attached:
            return
        previous = self.data.qpos[self.food_qpos : self.food_qpos + 3].copy()
        tcp_pos, tcp_rot = self._tcp_pose()
        position = tcp_pos + tcp_rot.apply(self.grasp_rel_pos)
        rotation = tcp_rot * self.grasp_rel_rot
        self.data.qpos[self.food_qpos : self.food_qpos + 3] = position
        quat = rotation.as_quat()
        self.data.qpos[self.food_qpos + 3 : self.food_qpos + 7] = (quat[3], quat[0], quat[1], quat[2])
        self.data.qvel[self.food_dof : self.food_dof + 3] = (position - previous) / dt
        self.data.qvel[self.food_dof + 3 : self.food_dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self, policy_action: np.ndarray) -> StepResult:
        policy_action = np.asarray(policy_action, dtype=np.float32).reshape(7)
        _, candidates = self._guidance()
        tcp_pos, _ = self._tcp_pose()
        policy_tensor = torch.from_numpy(policy_action).reshape(1, 7)
        tcp_tensor = torch.as_tensor(tcp_pos, dtype=torch.float32).reshape(1, 3)
        if self.controller_mode == "residual_baseline":
            finger_tensor = torch.as_tensor(
                self.data.qpos[self.finger_qpos], dtype=torch.float32
            ).reshape(1, 2)
            if self.pick_place:
                bowl_tensor = torch.as_tensor(
                    self.destination_bowl_pos, dtype=torch.float32
                ).reshape(1, 3)
                filtered, grasp_phase, phase = self.option(
                    policy_tensor, candidates, tcp_tensor, finger_tensor, bowl_tensor
                )
                phase_names = StableSpherePickPlaceSkill.PHASE_NAMES
                triggered = int(grasp_phase[0]) > StableSphereGraspSkill.HOVER
                grasp_phase_name = StableSphereGraspSkill.PHASE_NAMES[int(grasp_phase[0])]
            else:
                filtered, phase, _, _ = self.option(
                    policy_tensor, candidates, tcp_tensor, finger_tensor
                )
                phase_names = StableSphereGraspSkill.PHASE_NAMES
                triggered = int(phase[0]) > StableSphereGraspSkill.HOVER
                grasp_phase_name = phase_names[int(phase[0])]
        else:
            filtered, phase, _ = self.option(policy_tensor, candidates, tcp_tensor)
            phase_names = StableSphereCloseHoldLiftOption.PHASE_NAMES
            triggered = int(phase[0]) > StableSphereCloseHoldLiftOption.IDLE
            grasp_phase_name = phase_names[int(phase[0])]
        action = filtered[0].numpy().clip(-1.0, 1.0)
        processed = ACTION_SCALE * action[:6]
        tcp_pos, tcp_rot = self._tcp_pose()
        self.ee_target_pos = tcp_pos + processed[:3]
        self.ee_target_rot = Rotation.from_rotvec(processed[3:]) * tcp_rot
        finger_target = 0.02 + 0.02 * float(action[6])

        for substep in range(self.physics_steps):
            self._apply_ik()
            self.data.ctrl[self.finger_act] = finger_target
            target_t = self.elapsed + (substep + 1) * self.model.opt.timestep
            self.data.ctrl[self.pallet_act] = np.clip(
                self.pallet_initial_qpos + self.belt_speed * target_t,
                self.model.actuator_ctrlrange[self.pallet_act, 0],
                self.model.actuator_ctrlrange[self.pallet_act, 1],
            )
            arm_before = self.data.qpos[self.arm_qpos].copy()
            finger_before = self.data.qpos[self.finger_qpos].copy()
            mujoco.mj_step(self.model, self.data)
            arm_delta = np.clip(
                self.data.qpos[self.arm_qpos] - arm_before,
                -JOINT_VELOCITY * self.model.opt.timestep,
                JOINT_VELOCITY * self.model.opt.timestep,
            )
            self.data.qpos[self.arm_qpos] = np.clip(arm_before + arm_delta, JOINT_LOW, JOINT_HIGH)
            self.data.qvel[self.arm_dof] = arm_delta / self.model.opt.timestep
            finger_delta = np.clip(
                self.data.qpos[self.finger_qpos] - finger_before,
                -0.2 * self.model.opt.timestep,
                0.2 * self.model.opt.timestep,
            )
            self.data.qpos[self.finger_qpos] = np.clip(finger_before + finger_delta, 0.0, 0.04)
            self.data.qvel[self.finger_dof] = finger_delta / self.model.opt.timestep
            mujoco.mj_forward(self.model, self.data)
            self._update_contact_diagnostics(target_t)
            if finger_target < 0.039:
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

        sphere_x = float(self.data.xpos[self.food_body, 0])
        # With assistance disabled, strict success and the downstream deadline
        # must use real sustained bilateral contact rather than the (always
        # false) constraint-attachment flag.  This remains diagnostic state;
        # it is never exposed to the actor observation.
        physical_grasp_secured = self.grasp_attached or (
            not self.grasp_assist
            and self.diagnostics.current_dual_contact
            and self.diagnostics.max_dual_contact_run
            >= max(1, round(0.05 / self.model.opt.timestep))
        )
        sphere_position = self.data.xpos[self.food_body].copy()
        if self.pick_place:
            bowl_delta = sphere_position - self.destination_bowl_pos
            released = bool(np.all(self.data.qpos[self.finger_qpos] > 0.035))
            in_bowl = bool(
                np.linalg.norm(bowl_delta[:2]) < 0.10
                and 0.005 < bowl_delta[2] < 0.08
                and released
            )
            sphere_speed = float(np.linalg.norm(self.data.qvel[self.food_dof : self.food_dof + 3]))
            settled = in_bowl and sphere_speed < 0.05
            self._place_settle_count = self._place_settle_count + 1 if settled else 0
            settled_in_bowl = self._place_settle_count >= 5
            outcome = moving_sphere_place_outcome(
                settled_in_bowl=settled_in_bowl,
                grasp_ever_secured=self.diagnostics.ever_dual_contact or self.grasp_attached,
                sphere_x=sphere_x,
                sphere_z=float(sphere_position[2]),
                elapsed=self.elapsed,
                max_seconds=self.max_seconds,
            )
        else:
            bowl_delta = sphere_position - self.destination_bowl_pos
            in_bowl = False
            settled_in_bowl = False
            sphere_speed = float(np.linalg.norm(self.data.qvel[self.food_dof : self.food_dof + 3]))
            outcome = moving_sphere_outcome(
                grasp_attached=physical_grasp_secured,
                lift=lift,
                sphere_x=sphere_x,
                sphere_z=float(sphere_position[2]),
                elapsed=self.elapsed,
                max_seconds=self.max_seconds,
            )
        info: dict[str, float | bool | str] = {
            "time": round(self.elapsed, 4),
            "sphere_x": sphere_x,
            "sphere_z": float(self.data.xpos[self.food_body, 2]),
            "tcp_distance": translation,
            "rotation_error": rotation,
            "controller": self.controller_mode,
            "pick_place": self.pick_place,
            "controller_lift_height": (
                self.controller_lift_height if self.controller_mode == "residual_baseline" else 0.04
            ),
            "option_phase": phase_names[int(phase[0])],
            "grasp_phase": grasp_phase_name,
            "option_triggered": self.diagnostics.option_triggered,
            "grasp_assist_attached": self.grasp_attached,
            "physical_grasp_secured": physical_grasp_secured,
            "max_lift": self.diagnostics.max_lift,
            "ever_left_contact": self.diagnostics.ever_left_contact,
            "ever_right_contact": self.diagnostics.ever_right_contact,
            "ever_dual_contact": self.diagnostics.ever_dual_contact,
            "current_left_contact": self.diagnostics.current_left_contact,
            "current_right_contact": self.diagnostics.current_right_contact,
            "current_dual_contact": self.diagnostics.current_dual_contact,
            "first_dual_contact_time": self.diagnostics.first_dual_contact_time,
            "dual_contact_seconds": (
                self.diagnostics.dual_contact_substeps * self.model.opt.timestep
            ),
            "max_dual_contact_seconds": (
                self.diagnostics.max_dual_contact_run * self.model.opt.timestep
            ),
            "min_finger_gap": self.diagnostics.min_finger_gap,
            "sphere_speed": sphere_speed,
            "food_to_bowl_xy": float(np.linalg.norm(bowl_delta[:2])),
            "in_bowl": in_bowl,
            "settled_in_bowl": settled_in_bowl,
            "place_settle_steps": self._place_settle_count,
        }
        return StepResult(obs, outcome is not None, outcome, info)

    def close(self) -> None:
        pass
