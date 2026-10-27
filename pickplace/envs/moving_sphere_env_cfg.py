"""Moving-sphere/static-bowl task variants.

The control, observation, reward, and termination contracts intentionally
match the independently developed moving-cube task.  Only the food asset is
changed, which keeps historical cube checkpoints/results intact while giving
the real-sphere project its own registered task identity.
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from pickplace.envs import mdp
from pickplace.envs.moving_cube_env_cfg import (
    MovingCubeLiftEnvCfg,
    MovingCubeStaticBowlEnvCfg,
    MovingCubeTerminationsCfg,
)
from pickplace.food import ConveyorSphereCfg


@configclass
class MovingSphereStaticBowlEnvCfg(MovingCubeStaticBowlEnvCfg):
    """Franka intercepts a moving 40 mm sphere and places it in a fixed bowl."""

    food: ConveyorSphereCfg = ConveyorSphereCfg()

    def __post_init__(self):
        super().__post_init__()
        if self.grasp_point_observation:
            # The parent moving-cube task installs a 4x16 cuboid template.
            # Full sphere placement must preserve the accepted G0 checkpoint's
            # canonical 40x20 points plus 6D pose-delta observation contract.
            self.observations.grasp_guidance = SphereCanonicalGraspGuidanceCfg()


@configclass
class MovingSphereLiftEnvCfg(MovingCubeLiftEnvCfg):
    """Grasp-and-lift curriculum for the moving 40 mm sphere."""

    food: ConveyorSphereCfg = ConveyorSphereCfg()


@configclass
class SphereGapGraspGuidanceCfg(ObsGroup):
    """Deployable geometry only: 40x20 points plus nearest 6D pose error."""

    points_ee = ObsTerm(
        func=mdp.sphere_grasp_candidate_points_ee,
        params={
            "num_candidates": 40,
            "points_per_candidate": 20,
            "tcp_height_offset": 0.002,
            "grasp_width": 0.044,
        },
    )
    best_pose_delta = ObsTerm(
        func=mdp.sphere_best_grasp_pose_delta,
        params={"num_candidates": 40, "tcp_height_offset": 0.002, "rotation_weight": 0.05},
    )
    candidate_pose_deltas = ObsTerm(
        func=mdp.sphere_grasp_candidate_pose_deltas,
        params={"num_candidates": 40, "tcp_height_offset": 0.002},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class SphereCanonicalGraspGuidanceCfg(ObsGroup):
    """Symmetry-aware sphere grasp contract with one repeated canonical yaw.

    The tensor shapes intentionally match :class:`SphereGapGraspGuidanceCfg`
    so the point encoder and future camera interface remain unchanged.
    """

    points_ee = ObsTerm(
        func=mdp.sphere_grasp_candidate_points_ee,
        params={
            "num_candidates": 40,
            "points_per_candidate": 20,
            "tcp_height_offset": 0.002,
            "canonical_yaw": 0.0,
            "grasp_width": 0.044,
        },
    )
    best_pose_delta = ObsTerm(
        func=mdp.sphere_best_grasp_pose_delta,
        params={
            "num_candidates": 40,
            "tcp_height_offset": 0.002,
            "canonical_yaw": 0.0,
            "rotation_weight": 0.05,
        },
    )
    candidate_pose_deltas = ObsTerm(
        func=mdp.sphere_grasp_candidate_pose_deltas,
        params={"num_candidates": 40, "tcp_height_offset": 0.002, "canonical_yaw": 0.0},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class MovingSphereGapLiftEnvCfg(MovingSphereLiftEnvCfg):
    """Independent GAP-style sphere intercept/grasp/strict-3-cm-lift task."""

    target_lift: float = 0.03
    contact_force_threshold: float = 0.05
    contact_success_steps: int = 2
    stable_hold_steps: int = 5
    prelift_max_height: float = 0.005
    max_relative_drift: float = 0.008

    def __post_init__(self):
        if not self.speed_free_observations:
            raise ValueError("sphere_gap requires speed_free_observations=true")
        if not self.grasp_point_observation:
            raise ValueError("sphere_gap requires grasp_point_observation=true")
        if self.privileged_information:
            raise ValueError("sphere_gap actor observations must not include privileged/contact state")
        if self.gripper_action_mode != "continuous":
            raise ValueError("sphere_gap requires continuous gripper position control")
        if abs(float(self.target_lift) - 0.03) > 1.0e-9:
            raise ValueError("sphere_gap acceptance is fixed to a true 0.03 m lift")
        super().__post_init__()
        # MovingCubeStaticBowl installs the legacy four-region group. Replace it
        # after all parent builders ran; manager instantiation happens later.
        self.observations.grasp_guidance = SphereGapGraspGuidanceCfg()

    def _build_scene(self, zone) -> None:
        super()._build_scene(zone)
        if len(self.arm.gripper_body_names) != 2:
            raise ValueError("sphere_gap requires exactly two configured gripper bodies")
        finger_prims = self.arm.gripper_body_prim_paths or self.arm.gripper_body_names
        if len(finger_prims) != 2:
            raise ValueError("sphere_gap requires exactly two finger prim paths")
        self.scene.robot.spawn.activate_contact_sensors = True
        common = dict(
            update_period=0.0,
            history_length=1,
            track_air_time=False,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Food"],
        )
        self.scene.left_finger_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{finger_prims[0]}", **common
        )
        self.scene.right_finger_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{finger_prims[1]}", **common
        )

    def _build_belt_terms(self, zone) -> None:
        super()._build_belt_terms(zone)
        # For this lift-only stage the downstream edge is an interception
        # deadline, not a workspace boundary.  Once the sphere is visibly off
        # the belt, let the close/retract option finish and keep the independent
        # contact + true-3-cm-lift termination as the sole success criterion.
        self.terminations.cube_exited_zone.func = mdp.asset_exited_zone_unless_dual_contact
        self.terminations.cube_exited_zone.params = {
            "zone_end_x": zone.end_x,
            "ignore_above_z": self.belt.plate_top_z + self.food.item_radius + 0.005,
            "force_threshold": self.contact_force_threshold,
            "manipulation_distance": 0.05,
            "left_sensor_cfg": SceneEntityCfg("left_finger_contact"),
            "right_sensor_cfg": SceneEntityCfg("right_finger_contact"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "asset_cfg": SceneEntityCfg("food"),
        }

    def _build_food_terms(self) -> None:
        super()._build_food_terms()
        contact_params = {
            "force_threshold": self.contact_force_threshold,
            "left_sensor_cfg": SceneEntityCfg("left_finger_contact"),
            "right_sensor_cfg": SceneEntityCfg("right_finger_contact"),
        }
        self.rewards.reach_food.func = mdp.sphere_intercept_pose_approach
        self.rewards.reach_food.params = {
            "intercept_x": 0.45,
            "intercept_lead_distance": 0.08,
            "hover_height": 0.08,
            "transition_width": 0.01,
            "num_candidates": 40,
            "tcp_height_offset": 0.002,
            "food_cfg": SceneEntityCfg("food"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        }
        self.rewards.grasp.func = mdp.dual_finger_grasp
        self.rewards.grasp.params = dict(contact_params)
        self.rewards.grasp_lift.func = mdp.coupled_contact_lift_progress
        self.rewards.grasp_lift.params = {
            "surface_z": self.belt.plate_top_z,
            "item_radius": self.food.item_radius,
            "target_lift": 0.03,
            "stable_hold_steps": self.stable_hold_steps,
            "prelift_max_height": self.prelift_max_height,
            "max_relative_drift": self.max_relative_drift,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "food_cfg": SceneEntityCfg("food"),
            **contact_params,
        }
        self.terminations.success.func = mdp.coupled_contact_asset_lifted
        self.terminations.success.params = {
            "surface_z": self.belt.plate_top_z,
            "item_radius": self.food.item_radius,
            "target_lift": 0.03,
            "stable_hold_steps": self.stable_hold_steps,
            "success_steps": self.contact_success_steps,
            "prelift_max_height": self.prelift_max_height,
            "max_relative_drift": self.max_relative_drift,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "asset_cfg": SceneEntityCfg("food"),
            **contact_params,
        }


@configclass
class MovingSphereCanonicalGapLiftEnvCfg(MovingSphereGapLiftEnvCfg):
    """G0-S0 task with sphere yaw quotiented out of every grasp target.

    All 40 slots repeat one robot-friendly top-down grasp frame.  This keeps
    the GAP-style tensor contract while making the reward, actor guidance and
    stable option consume exactly the same physically meaningful target.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.grasp_guidance = SphereCanonicalGraspGuidanceCfg()

    def _build_food_terms(self) -> None:
        super()._build_food_terms()
        self.rewards.reach_food.params.update(
            {
                "canonical_yaw": 0.0,
                "trigger_translation": 0.018,
                "rotation_threshold": 0.15,
                "trigger_bonus": 0.5,
            }
        )


@configclass
class MovingSphereResidualTerminationsCfg(MovingCubeTerminationsCfg):
    """Canonical lift outcomes plus explicit Cartesian escape/retreat failure."""

    tcp_escape = DoneTerm(func=mdp.tcp_workspace_or_retreat, params={})


@configclass
class MovingSphereResidualGapLiftEnvCfg(MovingSphereCanonicalGapLiftEnvCfg):
    """Geometry-baseline task with signed progress and bounded PPO residual."""

    terminations: MovingSphereResidualTerminationsCfg = MovingSphereResidualTerminationsCfg()

    def _build_food_terms(self) -> None:
        super()._build_food_terms()
        common = {
            "intercept_x": 0.45,
            "intercept_lead_distance": 0.08,
            "hover_height": 0.08,
            "num_candidates": 40,
            "tcp_height_offset": 0.002,
            "canonical_yaw": 0.0,
            "food_cfg": SceneEntityCfg("food"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        }
        self.rewards.reach_food.func = mdp.sphere_residual_intercept_approach
        self.rewards.reach_food.params = {**common, "progress_gain": 20.0}
        self.terminations.tcp_escape.params = {
            **common,
            "workspace_lower": (0.10, -0.20, 0.02),
            "workspace_upper": (0.75, 0.48, 0.75),
            "retreat_epsilon": 0.0005,
            "retreat_steps": 15,
            "retreat_min_distance": 0.05,
        }


@configclass
class MovingSphereA0LiftEnvCfg(MovingSphereGapLiftEnvCfg):
    """Four-env learnability gate: physical near-grasp reset and strict lift.

    This task is deliberately Franka-specific and independent of the normal
    sphere curriculum.  Its default joint pose was measured from a successful
    closed-loop rollout at TCP=(0.3798, 0.2980, 0.1297) m over a sphere at
    (0.3906, 0.2976, 0.0500) m.  The sphere remains on the moving pallet; only
    the curriculum reset distribution is narrowed by its training config.
    """

    pregrasp_joint_pos: tuple[float, ...] = (
        -1.55131948,
        -0.25804791,
        2.25832868,
        -2.45404768,
        0.38310212,
        2.59809756,
        1.02313423,
    )

    def _build_scene(self, zone) -> None:
        super()._build_scene(zone)
        if len(self.pregrasp_joint_pos) != 7 or self.arm.ee_body_name != "panda_hand":
            raise ValueError("MovingSphereA0LiftEnvCfg is calibrated only for the 7-DoF Franka Panda")
        self.scene.robot.init_state.joint_pos.update(
            {f"panda_joint{index + 1}": value for index, value in enumerate(self.pregrasp_joint_pos)}
        )

    def _build_food_terms(self) -> None:
        super()._build_food_terms()
        self.rewards.reach_food.func = mdp.sphere_grasp_pose_approach_multiscale
        self.rewards.reach_food.params = {
            "translation_near_scale": 0.05,
            "translation_far_scale": 0.35,
            "rotation_near_scale": 0.35,
            "rotation_far_scale": 1.5,
            "near_weight": 0.3,
            "num_candidates": 40,
            "tcp_height_offset": 0.002,
            "food_cfg": SceneEntityCfg("food"),
        }
