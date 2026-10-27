"""Independent task: pick a moving cube from the conveyor and place it in a static bowl."""

from __future__ import annotations

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from pickplace.assets.scene_assets import make_bowl_cfg
from pickplace.belt import BeltCfg
from pickplace.envs import mdp
from pickplace.envs.cell_env_cfg import FoodCellEnvCfg, ObservationsCfg, TerminationsCfg
from pickplace.food import RigidCubeCfg


@configclass
class MovingCubeBeltObsCfg(ObsGroup):
    """Task quantities that can have real-cell equivalents."""

    bowl_pos = ObsTerm(func=mdp.asset_pos_cell, params={"asset_cfg": SceneEntityCfg("bowl")})
    speed = ObsTerm(func=mdp.conveyor_speed, params={"belt_event_name": "reset_belt"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class MovingCubePrivilegedCfg(ObsGroup):
    food_pos = ObsTerm(func=mdp.asset_pos_cell, params={"asset_cfg": SceneEntityCfg("food")})
    food_lin_vel = ObsTerm(func=mdp.asset_lin_vel_cell, params={"asset_cfg": SceneEntityCfg("food")})
    is_grasped = ObsTerm(func=mdp.is_grasped, params={"robot_cfg": SceneEntityCfg("robot")})
    grasp_proxy = None

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class MovingCubeTemporalCfg(ObsGroup):
    motion = ObsTerm(
        func=mdp.motion_history,
        params={
            "history_steps": 10,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class MovingCubePositionHistoryCfg(ObsGroup):
    position_action = ObsTerm(
        func=mdp.position_action_history,
        params={
            "history_steps": 10,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class MovingCubeGraspGuidanceCfg(ObsGroup):
    points_ee = ObsTerm(
        func=mdp.grasp_candidate_points_ee,
        params={"num_candidates": 4, "points_per_candidate": 16, "template_seed": 0},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class MovingCubeObservationsCfg(ObservationsCfg):
    belt: MovingCubeBeltObsCfg = MovingCubeBeltObsCfg()
    privileged: MovingCubePrivilegedCfg | None = MovingCubePrivilegedCfg()
    temporal: MovingCubeTemporalCfg | None = None
    position_history: MovingCubePositionHistoryCfg | None = None
    grasp_guidance: MovingCubeGraspGuidanceCfg | None = None


@configclass
class MovingCubeTerminationsCfg(TerminationsCfg):
    """Task-specific outcomes; the fixed bowl cannot exit, fall, or tip."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cube_exited_zone = DoneTerm(func=mdp.asset_exited_zone, params={})
    cube_off_belt = DoneTerm(func=mdp.asset_off_belt, params={})
    success = DoneTerm(func=mdp.food_in_bowl, params={})
    bowl_exited_zone = None
    bowl_off_belt = None
    bowl_tipped = None
    food_off_table = None


@configclass
class MovingCubeStaticBowlEnvCfg(FoodCellEnvCfg):
    """Franka intercepts a moving cube and deposits it into a fixed side bowl.

    Asset names intentionally remain ``food`` and ``bowl`` so the well-tested
    grasp, transport, placement, and Home terms can be reused. Their physical
    roles are changed without changing the original ``FoodCellEnvCfg``.
    """

    food: RigidCubeCfg = RigidCubeCfg()
    belt: BeltCfg = BeltCfg(speed=0.10, place_window=4.0)
    destination_bowl_pos: tuple[float, float, float] = (0.45, -0.10, 0.0)
    destination_bowl_x_range: tuple[float, float] = (0.45, 0.45)
    destination_bowl_y_range: tuple[float, float] = (-0.10, -0.10)

    observations: MovingCubeObservationsCfg = MovingCubeObservationsCfg()
    terminations: MovingCubeTerminationsCfg = MovingCubeTerminationsCfg()

    def __post_init__(self):
        # Parent validation/reset fields describe the old supply bowl. In this
        # task that same fixed-position facility resets the destination bowl.
        self.ingredient_bowl_pos = self.destination_bowl_pos
        self.ingredient_bowl_x_range = self.destination_bowl_x_range
        self.ingredient_bowl_y_range = self.destination_bowl_y_range
        if self.temporal_history_steps > 0:
            self.observations.temporal = MovingCubeTemporalCfg()
            self.observations.temporal.motion.params.update(
                history_steps=self.temporal_history_steps,
                robot_cfg=SceneEntityCfg("robot", joint_names=self.arm.gripper_joint_names),
            )
        if self.position_history_steps > 0:
            self.observations.position_history = MovingCubePositionHistoryCfg()
            self.observations.position_history.position_action.params.update(
                history_steps=self.position_history_steps,
                robot_cfg=SceneEntityCfg("robot", joint_names=self.arm.gripper_joint_names),
            )
        if self.grasp_point_observation:
            self.observations.grasp_guidance = MovingCubeGraspGuidanceCfg()
        if self.kinematic_grasp_proxy_observation:
            self.observations.privileged.grasp_proxy = ObsTerm(
                func=mdp.kinematic_grasp_proxy,
                params={
                    "robot_cfg": SceneEntityCfg("robot", joint_names=self.arm.gripper_joint_names),
                    "open_pos": self.arm.gripper_open,
                    "closed_pos": self.arm.gripper_closed,
                },
            )
        super().__post_init__()

    def _build_observations(self) -> None:
        super()._build_observations()
        if self.speed_free_observations:
            self.observations.proprio.joint_vel_rel = None
            self.observations.belt.speed = None
            if self.observations.privileged is not None:
                self.observations.privileged.food_lin_vel = None
                self.observations.privileged.food_pos = None
                self.observations.privileged.is_grasped = None
                self.observations.privileged.grasp_proxy = None
                if hasattr(self.observations.privileged, "food_quat"):
                    self.observations.privileged.food_quat = None

    def _build_scene(self, zone) -> None:
        super()._build_scene(zone)
        belt, scene = self.belt, self.scene
        # Replace the moving free bowl by the fixed destination bowl and remove
        # the original supply container. The pallet now carries the cube.
        scene.bowl = make_bowl_cfg(
            belt.bowl,
            "{ENV_REGEX_NS}/Bowl",
            self.destination_bowl_pos,
            kinematic=True,
        )
        scene.ingredient_bowl = None
        initial_food_pos = (
            belt.entry_x(),
            belt.belt_y,
            belt.plate_top_z + self.food.item_radius + 0.002,
        )
        scene.food = self.food.asset.replace(
            prim_path="{ENV_REGEX_NS}/Food",
            init_state=self.food.asset.init_state.replace(pos=initial_food_pos),
        )

    def _build_belt_terms(self, zone) -> None:
        belt = self.belt
        self.events.reset_belt.func = mdp.reset_food_on_belt
        self.events.reset_belt.params = {
            "speed": belt.speed,
            "speed_noise": belt.speed_noise,
            "pallet_start_range": belt.pallet_start_range,
            "food_offset_x": belt.bowl_offset_x,
            "food_offset_y": belt.bowl_offset_y,
            "entry_x": belt.entry_x(),
            "belt_y": belt.belt_y,
            "plate_top_z": belt.plate_top_z,
            "item_half_height": self.food.item_radius,
            "pallet_cfg": SceneEntityCfg("pallet"),
            "food_cfg": SceneEntityCfg("food"),
        }
        self.terminations.cube_exited_zone.params = {
            "zone_end_x": zone.end_x,
            "ignore_above_z": belt.plate_top_z + 0.08,
            "asset_cfg": SceneEntityCfg("food"),
        }
        self.terminations.cube_off_belt.params = {
            "belt_y": belt.belt_y,
            "belt_half_width": belt.belt_half_width,
            "surface_z": belt.plate_top_z,
            "lateral_check_max_z": belt.plate_top_z + 0.08,
            "lateral_safe_radius": belt.bowl.inner_radius,
            "lateral_safe_cfg": SceneEntityCfg("bowl"),
            "asset_cfg": SceneEntityCfg("food"),
        }

    def _build_food_terms(self) -> None:
        super()._build_food_terms()
        # Repoint the inherited fixed-container reset to the destination bowl.
        self.events.reset_ingredient_bowl.params.update(
            x_range=self.destination_bowl_x_range,
            y_range=self.destination_bowl_y_range,
            z=self.destination_bowl_pos[2],
            bowl_cfg=SceneEntityCfg("bowl"),
        )
        # A kinematic target bowl cannot be disturbed. Keep a zero component so
        # the shared 13-term dataset schema remains backward compatible.
        self.rewards.bowl_disturbance.func = mdp.zero
        self.rewards.bowl_disturbance.params = {}
        self.rewards.bowl_failure.params = {"term_names": []}
        self.rewards.food_dropped.params = {"term_names": ["cube_off_belt"]}


@configclass
class MovingCubeLiftEnvCfg(MovingCubeStaticBowlEnvCfg):
    """C0 curriculum: intercept, grasp, and lift the moving cube by 10 cm."""

    target_lift: float = 0.10
    stable_grasp_steps: int = 0
    stable_grasp_reward: float = 3.0
    grasp_slip_penalty: float = 0.0
    stable_prelift_max_height: float | None = None
    premature_lift_penalty: float = 0.0
    require_stable_grasp_success: bool = False
    geometry_only_grasp: bool = False
    geometric_slip_penalty: float = 0.0
    geometric_success_steps: int = 2
    geometric_close_success_steps: int = 0
    geometric_close_distance: float = 0.045
    geometric_close_min_fraction: float = 0.15
    geometric_close_max_fraction: float = 0.90
    geometric_far_close_penalty: float = 0.0
    geometric_bridge_enabled: bool = False
    geometric_bridge_progress_reward: bool = False
    geometric_bridge_close_steps: int = 5
    geometric_bridge_distance: float = 0.045
    geometric_bridge_min_fraction: float = 0.15
    geometric_bridge_max_fraction: float = 0.90
    geometric_bridge_prelift_max_height: float = 0.003
    geometric_bridge_target_lift: float = 0.005
    geometric_bridge_max_relative_drift: float = 0.008
    geometric_bridge_success_steps: int = 5
    geometric_bridge_tcp_progress_weight: float = 0.25
    geometric_bridge_object_progress_weight: float = 1.0
    geometric_stable_steps: int = 0
    geometric_stable_distance: float = 0.04
    geometric_stable_relative_speed: float = 0.035
    geometric_stable_distance_std: float = 0.04
    geometric_stable_relative_speed_std: float = 0.04
    geometric_stable_hold_reward: float = 1.0
    geometric_premature_close_penalty: float = 0.25
    geometric_confidence_loss_penalty: float = 1.0
    geometric_lift_velocity_weight: float = 0.0
    geometric_lift_velocity_std: float = 0.05

    def _build_food_terms(self) -> None:
        super()._build_food_terms()
        arm, belt = self.arm, self.belt
        # Return fresh SceneEntityCfg objects for every manager term. Resolution
        # mutates them in place (joint_names -> joint_ids), so sharing one
        # between the stateful grasp and lift terms makes the second resolution
        # fail even though both select the same finger joints.
        def grasp_params() -> dict:
            return {
                "robot_cfg": SceneEntityCfg("robot", joint_names=arm.gripper_joint_names),
                "open_pos": arm.gripper_open,
                "closed_pos": arm.gripper_closed,
            }

        def lift_params() -> dict:
            return {
                "surface_z": belt.plate_top_z,
                "item_radius": self.food.item_radius,
                "target_lift": self.target_lift,
                "food_cfg": SceneEntityCfg("food"),
                **grasp_params(),
            }
        if self.geometry_only_grasp:
            if self.geometric_stable_steps > 0:
                self.rewards.grasp.func = mdp.geometric_stable_close
                self.rewards.grasp.params = {
                    "stable_steps": self.geometric_stable_steps,
                    "distance_threshold": self.geometric_stable_distance,
                    "relative_speed_threshold": self.geometric_stable_relative_speed,
                    "distance_std": self.geometric_stable_distance_std,
                    "relative_speed_std": self.geometric_stable_relative_speed_std,
                    "hold_reward": self.geometric_stable_hold_reward,
                    "premature_close_penalty": self.geometric_premature_close_penalty,
                    "confidence_loss_penalty": self.geometric_confidence_loss_penalty,
                    "food_cfg": SceneEntityCfg("food"),
                    **grasp_params(),
                }
            else:
                self.rewards.grasp.func = mdp.geometric_close_alignment
                self.rewards.grasp.params = {
                    "std": 0.04,
                    "far_close_penalty": self.geometric_far_close_penalty,
                    "food_cfg": SceneEntityCfg("food"),
                    **grasp_params(),
                }
            self.rewards.grasp_lift.func = mdp.geometric_lift_progress
            self.rewards.grasp_lift.params = {
                "surface_z": belt.plate_top_z,
                "item_radius": self.food.item_radius,
                "target_lift": self.target_lift,
                "coupling_std": 0.05,
                "slip_min_lift": 0.005,
                "slip_distance": 0.075,
                "slip_fall": 0.008,
                "slip_penalty": self.geometric_slip_penalty,
                "relative_speed_weight": self.geometric_lift_velocity_weight,
                "relative_speed_std": self.geometric_lift_velocity_std,
                "food_cfg": SceneEntityCfg("food"),
            }
        elif self.stable_grasp_steps > 0:
            # Keep the shared reward-vector schema: the existing ``grasp``
            # column now carries consecutive-hold shaping plus a one-shot slip
            # penalty, and ``grasp_lift`` is gated by the same hold duration.
            self.rewards.grasp.func = mdp.stable_grasp_stage
            self.rewards.grasp.params = {
                "stable_steps": self.stable_grasp_steps,
                "hold_reward": self.stable_grasp_reward,
                "slip_penalty": self.grasp_slip_penalty,
                "food_cfg": SceneEntityCfg("food"),
                **grasp_params(),
            }
            self.rewards.grasp_lift.func = mdp.stable_grasp_lift_progress
            self.rewards.grasp_lift.params = {
                "stable_steps": self.stable_grasp_steps,
                "prelift_max_height": self.stable_prelift_max_height,
                "premature_lift_penalty": self.premature_lift_penalty,
                **lift_params(),
            }
        else:
            self.rewards.grasp_lift.func = mdp.grasp_lift_progress
            self.rewards.grasp_lift.params = lift_params()
        if self.geometry_only_grasp:
            self.terminations.success.func = mdp.geometric_asset_lifted
            self.terminations.success.params = {
                "surface_z": belt.plate_top_z,
                "item_radius": self.food.item_radius,
                "target_lift": self.target_lift,
                "max_tcp_distance": 0.075,
                "stable_steps": self.geometric_success_steps,
                "asset_cfg": SceneEntityCfg("food"),
            }
        else:
            self.terminations.success.func = mdp.asset_lifted
            self.terminations.success.params = {
                "surface_z": belt.plate_top_z,
                "item_radius": self.food.item_radius,
                "target_lift": self.target_lift,
                "asset_cfg": SceneEntityCfg("food"),
                **grasp_params(),
            }
        if self.geometric_close_success_steps > 0:
            self.terminations.success.func = mdp.geometric_sustained_close
            self.terminations.success.params = {
                "distance_threshold": self.geometric_close_distance,
                "min_close_fraction": self.geometric_close_min_fraction,
                "max_close_fraction": self.geometric_close_max_fraction,
                "stable_steps": self.geometric_close_success_steps,
                "asset_cfg": SceneEntityCfg("food"),
                **grasp_params(),
            }
        if self.geometric_bridge_enabled:
            self.terminations.success.func = mdp.geometric_close_then_lift
            self.terminations.success.params = {
                "surface_z": belt.plate_top_z,
                "item_radius": self.food.item_radius,
                "close_steps": self.geometric_bridge_close_steps,
                "distance_threshold": self.geometric_bridge_distance,
                "min_close_fraction": self.geometric_bridge_min_fraction,
                "max_close_fraction": self.geometric_bridge_max_fraction,
                "prelift_max_height": self.geometric_bridge_prelift_max_height,
                "target_lift": self.geometric_bridge_target_lift,
                "max_relative_drift": self.geometric_bridge_max_relative_drift,
                "success_steps": self.geometric_bridge_success_steps,
                "asset_cfg": SceneEntityCfg("food"),
                **grasp_params(),
            }
            if self.geometric_bridge_progress_reward:
                self.rewards.grasp_lift.func = mdp.geometric_bridge_lift_progress
                self.rewards.grasp_lift.params = {
                    "surface_z": belt.plate_top_z,
                    "item_radius": self.food.item_radius,
                    "close_steps": self.geometric_bridge_close_steps,
                    "distance_threshold": self.geometric_bridge_distance,
                    "min_close_fraction": self.geometric_bridge_min_fraction,
                    "max_close_fraction": self.geometric_bridge_max_fraction,
                    "prelift_max_height": self.geometric_bridge_prelift_max_height,
                    "target_lift": self.geometric_bridge_target_lift,
                    "max_relative_drift": self.geometric_bridge_max_relative_drift,
                    "tcp_progress_weight": self.geometric_bridge_tcp_progress_weight,
                    "object_progress_weight": self.geometric_bridge_object_progress_weight,
                    "food_cfg": SceneEntityCfg("food"),
                    **grasp_params(),
                }
        if self.require_stable_grasp_success:
            if self.stable_prelift_max_height is None:
                raise ValueError("require_stable_grasp_success needs stable_prelift_max_height")
            self.terminations.success.func = mdp.stable_grasp_asset_lifted
            self.terminations.success.params.update(
                stable_steps=self.stable_grasp_steps,
                prelift_max_height=self.stable_prelift_max_height,
            )
