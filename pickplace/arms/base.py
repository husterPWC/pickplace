# pickplace/arms/base.py
"""Arm plug-in configuration: everything the env needs to know about a manipulator."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass


@configclass
class ArmCfg:
    robot: ArticulationCfg = MISSING
    """Articulation used with joint-space actions (prim_path is overwritten by the env)."""
    ik_robot: ArticulationCfg = MISSING
    """Articulation with stiffer gains used with differential-IK actions."""
    arm_joint_names: list[str] = MISSING
    policy_joint_names: list[str] | None = None
    """Optional joints exposed by ``joint_pos_rel``/``joint_vel_rel``.

    Leave unset to preserve the historical all-joint Franka observation.  An
    imported linkage gripper may use this to hide simulator-only follower
    coordinates; its real encoder remains available through ``gripper_pos``.
    """
    gripper_joint_names: list[str] = MISSING
    gripper_control_joint_names: list[str] | None = None
    """Optional physical joints driven by the one-dimensional gripper action.

    ``gripper_joint_names`` remains the logical encoder coordinate exposed to
    observations and grasp predicates.  This separate list supports imported
    linkage grippers whose PhysX USD does not enforce mimic metadata: one
    policy coordinate can still synchronously command every follower joint.
    """
    gripper_body_names: list[str] = MISSING
    """Rigid body names the fingers spawn as, used to apply ``finger_friction`` (e.g. via a startup
    ``randomize_rigid_body_material`` event); distinct from ``gripper_joint_names``, which drives the gripper
    action and observations."""
    base_link_name: str = MISSING
    ee_body_name: str = MISSING
    base_link_prim_path: str | None = None
    """Optional path below ``Robot/`` when the USD preserves a nested link hierarchy."""
    ee_body_prim_path: str | None = None
    """Optional path below ``Robot/`` for frame sensors and wrist-camera parenting."""
    gripper_body_prim_paths: list[str] | None = None
    """Optional nested paths below ``Robot/`` for per-finger contact sensors."""
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Tool center point offset from the end-effector body frame."""
    tcp_rot_xyzw: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """TCP orientation relative to the end-effector body in this build's xyzw convention.

    The default deliberately preserves the historical Franka offset produced
    by this IsaacLab fork (its config docstring still says wxyz although the
    runtime math is xyzw).  A distinct offset is required for AG95 because its
    body axes do not use the policy convention (+y closing, +z approach).
    """
    ik_uses_tcp_offset: bool = True
    """Whether differential IK applies the configured TCP offset.

    Disable this for long, rotated tools when controlling the coincident
    flange origin is numerically better conditioned.  Frame observations and
    camera parenting still use the physical TCP offset.
    """
    persistent_ik_target: bool = False
    """Integrate relative pose commands from the last command instead of re-latching measured pose."""
    home_tcp_pos: tuple[float, float, float] = MISSING
    """TCP position (env-local, [m]) at the robot's default joint pose: "home" for ``success_requires_home`` and
    the ``return_home`` reward. Measured in the simulator with ``scripts/measure_home_tcp.py``."""
    gripper_open: float = MISSING
    gripper_closed: float = MISSING
    wrist_cam_offset: CameraCfg.OffsetCfg = MISSING
    """Wrist camera pose relative to the end-effector body."""
    reach_radius: float = MISSING
    """Conservative horizontal reach from the base, used to validate the belt zone."""
    joint_action_scale: float = 0.5
    ik_action_scale: float = 0.5
    finger_friction: tuple[float, float] = (1.2, 1.0)
    """(static, dynamic) friction applied to ``gripper_body_names`` by the env's ``gripper_material`` startup
    event. Without this, the fingers spawn with no material of their own and inherit PhysX's 0.5/0.5 default,
    which (combine mode unset, i.e. averaged against the food's material) is too slippery to lift a smooth
    sphere gripped by flat pads -- see task-13-grasp-fix.md."""
