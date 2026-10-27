"""JAKA-oriented static-sphere supply bowl -> moving destination bowl task."""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from pickplace.envs import mdp
from pickplace.envs.cell_env_cfg import FoodCellEnvCfg
from pickplace.envs.moving_cube_env_cfg import MovingCubePositionHistoryCfg


@configclass
class StaticSphereGraspGuidanceCfg(ObsGroup):
    """Canonical speed-free 40x20 sphere geometry for the state teacher."""

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
        params={
            "num_candidates": 40,
            "tcp_height_offset": 0.002,
            "canonical_yaw": 0.0,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class StaticSphereMovingBowlEnvCfg(FoodCellEnvCfg):
    """Static supported sphere, 0.10 m/s destination bowl, and return Home."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.table = self.scene.table.replace(
            spawn=self.scene.table.spawn.replace(size=(1.50, 1.40, 0.80))
        )
        if self.grasp_point_observation:
            self.observations.grasp_guidance = StaticSphereGraspGuidanceCfg()
        if self.position_history_steps > 0:
            self.observations.position_history = MovingCubePositionHistoryCfg()
            self.observations.position_history.position_action.params.update(
                history_steps=self.position_history_steps,
                robot_cfg=SceneEntityCfg(
                    "robot", joint_names=self.arm.gripper_joint_names
                ),
            )
