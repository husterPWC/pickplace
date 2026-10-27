"""JAKA ZU5 with the directly mounted DH-Robotics AG95 gripper."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.sim import schemas
from isaaclab.sim.utils import find_matching_prim_paths
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab_physx.sim.schemas import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg

from pickplace.arms.base import ArmCfg


_ROOT = Path(__file__).resolve().parents[2]
_USD = _ROOT / "assets/models/robots/Zu5Ag95/usd/zu5_ag95/zu5_ag95.usda"
_GRIPPER_JOINTS = [
    "left_outer_knuckle_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "left_finger_joint",
    "right_finger_joint",
]
_LEFT_FINGER_PRIM = (
    "Geometry/Link_0/Link_1/Link_2/Link_3/Link_4/Link_5/Link_6/"
    "tool0/ag95_base_link/left_outer_knuckle/left_finger"
)
_RIGHT_FINGER_PRIM = (
    "Geometry/Link_0/Link_1/Link_2/Link_3/Link_4/Link_5/Link_6/"
    "tool0/ag95_base_link/right_outer_knuckle/right_finger"
)

# Simulator-free IK solution for the conveyor standby TCP
# (0.45, 0.30, 0.134) m with the canonical top-down grasp orientation.
#
# This is deliberately the shoulder-away branch.  The more obvious solution
# [0.3756, -0.1872, 1.7778, ...] reaches the same TCP pose, but sweeps Link_2
# through the conveyor corridor and physically knocks the pallet/sphere before
# control starts.  This branch leaves 138 mm conservative AABB clearance from
# the incoming-object corridor.
_STANDBY = {
    "joint_1": -2.341213,
    "joint_2": 1.740149,
    "joint_3": 1.777842,
    "joint_4": 1.194395,
    "joint_5": 4.712393,
    "joint_6": -2.341218,
}


def _spawn_with_finger_contact_reporters(
    prim_path: str,
    cfg,
    translation=None,
    orientation=None,
    **kwargs,
):
    """Spawn the nested USD, then enable reports on the two leaf fingers.

    Isaac Lab's generic recursive helper stops at the first rigid body.  The
    converted multi-body USD preserves nested rigid links, so enabling at the
    articulation root reaches Link_0 but not either finger.
    """
    prim = sim_utils.spawn_from_usd(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )
    for suffix in (_LEFT_FINGER_PRIM, _RIGHT_FINGER_PRIM):
        matches = find_matching_prim_paths(f"{prim_path}/{suffix}")
        if not matches:
            raise RuntimeError(f"missing AG95 finger prim under {prim_path}: {suffix}")
        for path in matches:
            schemas.activate_contact_sensors(path, True)
    return prim


def _robot_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            func=_spawn_with_finger_contact_reporters,
            usd_path=str(_USD),
            activate_contact_sensors=False,
            articulation_props=ArticulationRootPropertiesCfg(
                # The converted asset already contains its fixed root joint.
                fix_root_link=None,
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                # Match Isaac Lab's high-PD manipulation assets: the real
                # robot controller performs gravity compensation.  Leaving
                # gravity on for every lightweight AG95 linkage body makes
                # this imported six-DOF linkage numerically drift despite
                # position drives; the scene objects retain normal gravity.
                disable_gravity=True,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={**_STANDBY, ".*(knuckle|finger)_joint": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={
            "zu5_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                stiffness=10000.0,
                damping=500.0,
                armature=0.1,
                effort_limit_sim=5000.0,
                velocity_limit_sim=3.14,
            ),
            # PhysX does not consume the converter's newton:mimicJoint
            # metadata.  Drive all linkage coordinates to the same target;
            # CoupledGripperPositionAction still exposes exactly one policy
            # action and therefore matches the real AG95 interface.
            "ag95_coupled": ImplicitActuatorCfg(
                joint_names_expr=_GRIPPER_JOINTS,
                stiffness=200.0,
                damping=20.0,
                armature=0.01,
                effort_limit_sim=200.0,
                velocity_limit_sim=2.1,
            ),
        },
    )


ZU5_AG95_CFG = ArmCfg(
    robot=_robot_cfg(),
    ik_robot=_robot_cfg(),
    arm_joint_names=["joint_[1-6]"],
    # The five additional AG95 linkage coordinates are simulator-only mimic
    # followers.  The actor receives the six ZU5 joints here and the one real
    # AG95 master encoder separately through ``gripper_pos``.
    policy_joint_names=["joint_[1-6]"],
    # One policy coordinate drives the master; five passive mimic joints form
    # the validated closed four-bar approximation.
    gripper_joint_names=["left_outer_knuckle_joint"],
    gripper_control_joint_names=_GRIPPER_JOINTS,
    gripper_body_names=["left_finger", "right_finger"],
    base_link_name="Link_0",
    # ag95_base_link is fixed to Link_6 at an identity transform.  PhysX keeps
    # the fixed child in body_names but omits it from the reduced Jacobian;
    # selecting it makes DifferentialIK index the following row and diverge.
    # Use its coincident actuated parent for both pose and Jacobian queries.
    ee_body_name="Link_6",
    base_link_prim_path="Geometry/Link_0",
    ee_body_prim_path="Geometry/Link_0/Link_1/Link_2/Link_3/Link_4/Link_5/Link_6",
    gripper_body_prim_paths=[
        _LEFT_FINGER_PRIM,
        _RIGHT_FINGER_PRIM,
    ],
    # The measured pad midpoint is 167.56 mm along AG95 +z.  Rotate the policy
    # frame -90 deg around that axis so policy +y is the physical closing axis.
    tcp_offset=(0.0, 0.0, 0.16756),
    # Link_6 differs from the fixed AG95 base by a 180-degree frame flip;
    # this composes the Link_6 frame into the canonical top-down policy TCP.
    tcp_rot_xyzw=(0.0, 0.0, -0.7071067812, -0.7071067812),
    ik_uses_tcp_offset=True,
    persistent_ik_target=True,
    home_tcp_pos=(0.45, 0.30, 0.134),
    gripper_open=0.0,
    gripper_closed=0.943,
    # Placeholder physical mount used only when G2 cameras are enabled; the
    # high-driver camera smoke must validate and, if needed, recalibrate it.
    wrist_cam_offset=CameraCfg.OffsetCfg(
        pos=(0.06, 0.0, 0.10),
        rot=(0.0, -0.173648, 0.0, 0.984808),
        convention="ros",
    ),
    reach_radius=0.90,
    ik_action_scale=0.02,
    finger_friction=(1.5, 1.2),
)
