"""Build a FoodCellEnvCfg from a plain mapping (e.g. the ``env:`` section of a Hydra config)."""
from __future__ import annotations

import importlib
import copy
from collections.abc import Mapping

from pickplace.action_adapters import ADAPTERS

ARMS = {
    "franka": "pickplace.arms.franka:FRANKA_CFG",
    "zu5_ag95": "pickplace.arms.zu5_ag95:ZU5_AG95_CFG",
}
FOODS = {
    "rigid": "pickplace.food.rigid:RigidFoodCfg",
    "cube": "pickplace.food.cube:RigidCubeCfg",
    "moving_sphere": "pickplace.food.conveyor_sphere:ConveyorSphereCfg",
}

TASK_CONFIGS = {
    "FoodRobot-Cell-v0": "pickplace.envs.cell_env_cfg:FoodCellEnvCfg",
    "FoodRobot-MovingCubeStaticBowl-v0": "pickplace.envs.moving_cube_env_cfg:MovingCubeStaticBowlEnvCfg",
    "FoodRobot-MovingCubeLift-v0": "pickplace.envs.moving_cube_env_cfg:MovingCubeLiftEnvCfg",
    "FoodRobot-MovingSphereStaticBowl-v0": (
        "pickplace.envs.moving_sphere_env_cfg:MovingSphereStaticBowlEnvCfg"
    ),
    "FoodRobot-MovingSphereLift-v0": "pickplace.envs.moving_sphere_env_cfg:MovingSphereLiftEnvCfg",
    "FoodRobot-MovingSphereGapLift-v0": "pickplace.envs.moving_sphere_env_cfg:MovingSphereGapLiftEnvCfg",
    "FoodRobot-MovingSphereCanonicalGapLift-v0": (
        "pickplace.envs.moving_sphere_env_cfg:MovingSphereCanonicalGapLiftEnvCfg"
    ),
    "FoodRobot-MovingSphereA0Lift-v0": "pickplace.envs.moving_sphere_env_cfg:MovingSphereA0LiftEnvCfg",
    "FoodRobot-MovingSphereResidualGapLift-v0": (
        "pickplace.envs.moving_sphere_env_cfg:MovingSphereResidualGapLiftEnvCfg"
    ),
    "FoodRobot-StaticSphereMovingBowl-v0": (
        "pickplace.envs.static_sphere_moving_bowl_env_cfg:StaticSphereMovingBowlEnvCfg"
    ),
}

MOVING_STATIC_BOWL_TASKS = {
    "FoodRobot-MovingCubeStaticBowl-v0",
    "FoodRobot-MovingSphereStaticBowl-v0",
}
MOVING_LIFT_TASKS = {
    "FoodRobot-MovingCubeLift-v0",
    "FoodRobot-MovingSphereLift-v0",
    "FoodRobot-MovingSphereGapLift-v0",
    "FoodRobot-MovingSphereCanonicalGapLift-v0",
    "FoodRobot-MovingSphereA0Lift-v0",
    "FoodRobot-MovingSphereResidualGapLift-v0",
}


DEFAULT_ENV: dict = {
    "task": "FoodRobot-Cell-v0",
    "num_envs": 16,
    "arm": "franka",
    "food": "rigid",
    "action_mode": "ee_delta_pose",
    "gripper_action_mode": "binary",
    # None preserves the arm plug-in default (historically 0.5 m/rad).  The
    # independent sphere_gap route overrides this to 0.02 for centimetre-scale
    # Cartesian increments without modifying FRANKA_CFG or historical tasks.
    "ik_action_scale": None,
    # Action parameterization layered on top of action_mode by pickplace.action_adapters: "native" keeps the
    # env's own space, "joint_velocity" / "joint_position" convert at the boundary (both need joint_pos mode).
    "action_adapter": "native",
    "cameras": True,
    "camera_depth": False,
    "image_size": [128, 128],
    "frame_stack": 1,
    "privileged_information": False,
    "grasp_observation": True,
    "temporal_history_steps": 0,
    "kinematic_grasp_proxy_observation": False,
    "position_history_steps": 0,
    "grasp_point_observation": False,
    "speed_free_observations": False,
    "belt": {},
    # Reset randomization added to the arm joints' default pose (reset_joints_by_offset; see
    # FoodCellEnvCfg.robot_reset). Defaults are exactly today's values, so every existing config is unaffected.
    "robot_reset": {"position_range": [-0.02, 0.02], "velocity_range": [0.0, 0.0]},
    # Optional task-specific standby pose.  Keys are exact articulation joint
    # names and values are radians.  None preserves every historical task's
    # asset default; conveyor stages can start from a calibrated interception
    # posture instead of spending the object's flight time leaving Home.
    "robot_initial_joint_pos": None,
    "seed": 0,
    "device": "cuda:0",
    # Legacy one-shot keys: None defers to the reward set (see pickplace.rewards.resolve_reward_weights).
    "success_bonus": None,
    "bowl_failure_penalty": None,
    "food_drop_penalty": None,
    "success_settle_steps": 5,
    # Success also requires the TCP back within home_tolerance [m] of its home position (FoodCellEnvCfg).
    "success_requires_home": False,
    "home_tolerance": 0.05,
    "ingredient_bowl_pos": [0.45, -0.10, 0.0],
    # Fixed as of task 14 (was [0.35, 0.55] / [-0.20, 0.00]): the food is randomized inside the wider supply
    # tray instead (RigidFoodCfg.spawn_range), so the container itself no longer needs its own position DR.
    # Kept as a real range (not folded into a single value) so it can be widened again later.
    "ingredient_bowl_x_range": [0.45, 0.45],
    "ingredient_bowl_y_range": [-0.10, -0.10],
    # Used only by FoodRobot-MovingCubeStaticBowl-v0. Kept separate so the
    # original task's supply-bowl contract and old checkpoint configs do not change.
    "destination_bowl_pos": [0.45, -0.10, 0.0],
    "destination_bowl_x_range": [0.45, 0.45],
    "destination_bowl_y_range": [-0.10, -0.10],
    # Used only by FoodRobot-MovingCubeLift-v0.  Keeping this configurable lets
    # the lift recovery curriculum progress from a short, discoverable lift to
    # the full 10 cm clearance without introducing another Gym task.
    "target_lift": 0.10,
    "geometry_only_grasp": False,
    "geometric_slip_penalty": 0.0,
    "geometric_success_steps": 2,
    # Optional first curriculum stage: stable aligned closure is success.
    # Disabled by default to preserve all existing lift-task contracts.
    "geometric_close_curriculum": {
        "success_steps": 0,
        "distance_threshold": 0.045,
        "min_close_fraction": 0.15,
        "max_close_fraction": 0.90,
        "far_close_penalty": 0.0,
    },
    # Contact-free bridge between sustained closure and a full lift.  The
    # success term first latches a near-surface close, then requires the cube
    # to co-move with the TCP through a small vertical displacement.
    "geometric_close_lift_bridge": {
        "enabled": False,
        "progress_reward": False,
        "close_steps": 5,
        "distance_threshold": 0.045,
        "min_close_fraction": 0.15,
        "max_close_fraction": 0.90,
        "prelift_max_height": 0.003,
        "target_lift": 0.005,
        "max_relative_drift": 0.008,
        "success_steps": 5,
        "tcp_progress_weight": 0.25,
        "object_progress_weight": 1.0,
    },
    # Contact-free temporal grasp-target stabilisation.  stable_steps=0 keeps
    # every existing task and checkpoint reward contract unchanged.
    "geometric_grasp_stability": {
        "stable_steps": 0,
        "distance_threshold": 0.04,
        "relative_speed_threshold": 0.035,
        "distance_std": 0.04,
        "relative_speed_std": 0.04,
        "hold_reward": 1.0,
        "premature_close_penalty": 0.25,
        "confidence_loss_penalty": 1.0,
    },
    "geometric_lift_velocity": {
        "weight": 0.0,
        "std": 0.05,
    },
    # Moving-cube lift only. hold_steps=0 preserves the original stateless C0
    # reward exactly; a positive value enables the consecutive-hold curriculum.
    "grasp_stability": {
        "hold_steps": 0,
        "hold_reward": 3.0,
        "slip_penalty": 0.0,
        "prelift_max_height": None,
        "premature_lift_penalty": 0.0,
        "require_stable_success": False,
    },
    # Optional controller-boundary gate. Zero steps leaves policy actions
    # byte-for-byte unchanged; positive steps require privileged is_grasped.
    "grasp_hold_gate": {
        "hold_steps": 0,
        "max_upward_dz": 0.0,
        "force_close": True,
        "follow_object_xy": False,
        "follow_gain": 0.5,
        "max_follow_speed": 0.15,
        "freeze_rotation": False,
    },
    "close_retract_option": {
        "steps": 0,
        "retract_height": 0.035,
        "trigger_distance": 0.045,
        "trigger_gripper": -0.5,
        "follow_gain": 0.5,
        "max_follow_speed": 0.15,
        "xy_blend": 0.5,
    },
    "grasp_pose_close_retract_option": {
        "enabled": False,
        # Start the short visual-servo option with enough downstream margin
        # for the fastest S0a samples; alignment is refined while still open.
        "translation_threshold": 0.018,
        "rotation_threshold": 0.15,
        "approach_steps": 4,
        "close_steps": 12,
        "retract_steps": 15,
        "retract_height": 0.04,
        # Franka mapping: -0.15 -> 0.017 m per finger (~34 mm total
        # commanded opening).  Stronger closure increased lateral squeeze-out
        # in the deterministic physical probe.
        "close_action": -0.15,
        "position_gain": 1.5,
        "rotation_gain": 0.5,
        "max_position_step": 0.004,
        "max_rotation_step": 0.01,
    },
    "stable_sphere_grasp_option": {
        "enabled": False,
        "trigger_translation": 0.018,
        "grasp_translation": 0.007,
        "rotation_threshold": 0.15,
        "alignment_steps": 4,
        "close_steps": 16,
        "hold_steps": 10,
        "lift_steps": 30,
        "lift_timeout_steps": 150,
        "lift_height": 0.04,
        "lift_tolerance": 0.003,
        "align_position_gain": 4.0,
        "position_gain": 1.5,
        "rotation_gain": 0.5,
        "max_align_position_step": 0.008,
        "max_position_step": 0.004,
        "max_rotation_step": 0.01,
        "close_action": -0.15,
    },
    # Full validated geometry controller as the nominal command.  The policy
    # supplies only a small, normalized arm residual during hover/alignment;
    # close/hold/lift and all finger commands remain deterministic.
    "stable_sphere_residual_skill": {
        "enabled": False,
        "intercept_x": 0.45,
        "intercept_lead_distance": 0.08,
        "intercept_align_lead_distance": 0.04,
        "hover_height": 0.08,
        "pre_descent_height": None,
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
        "lift_target_slack": None,
        "lift_detach_height": None,
        "lift_xy_gain": None,
        "max_lift_xy_step": None,
        "preserve_lift_relative_xy": False,
        "target_xy_feedforward_gain": 0.0,
        "max_target_xy_feedforward_step": None,
        "align_target_xy_feedforward": False,
        "close_requires_raw_alignment": False,
        "max_close_hold_xy_step": None,
        "latch_lift_xy_feedforward": False,
        "lift_feedforward_x_only": False,
        "lift_feedforward_decay_steps": None,
        "lift_geometry_relatch_height": None,
        "lift_relative_z_gain": None,
        "settled_coupling_reference": False,
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
        "close_action": -0.15,
        "translation_alpha": 0.45,
        "rotation_alpha": 0.35,
        "max_translation_jump": 0.02,
        "max_rotation_jump": 0.35,
        "stable_translation": 0.004,
        "stable_rotation": 0.08,
        "stable_steps": 4,
        "rotation_weight": 0.05,
        "switch_margin": 0.01,
        "translation_residual_limit": 0.10,
        "rotation_residual_limit": 0.05,
    },
    # G1 wraps the validated G0 residual interceptor with deterministic
    # clearance, transport, descent, and release.  The actor retains bounded
    # authority only inside the G0 grasp controller; transport consumes the
    # calibrated/observed bowl pose but no velocity, contact, or force signal.
    "stable_sphere_pick_place_skill": {
        "enabled": False,
        "clear_height": 0.20,
        "bowl_hover_height": 0.20,
        "release_height": 0.11,
        "position_gain": 4.0,
        "max_position_step": 0.016,
        "max_translation_action_delta": 0.04,
        "max_lower_xy_step": 0.002,
        "position_deadband": 0.001,
        "clear_tolerance": 0.012,
        "hover_tolerance": 0.018,
        "release_tolerance": 0.012,
        "stable_steps": 3,
        "release_steps": 28,
        "close_action": -0.15,
        "open_action": 1.0,
        "home_position": None,
        "retreat_height": 0.20,
        "retreat_tolerance": 0.015,
        "home_tolerance": 0.025,
        "max_home_position_step": 0.016,
    },
    # Speed/contact/force-free bridge used after a sustained geometric close.
    # It reads only the position history, current TCP pose, and finger encoders.
    "geometric_micro_lift_option": {
        "enabled": False,
        "close_steps": 5,
        "lift_steps": 12,
        "hold_steps": 8,
        "lift_height": 0.008,
        "distance_threshold": 0.045,
        "min_close_fraction": 0.15,
        "max_close_fraction": 0.90,
        "xy_gain": 0.5,
        "z_gain": 1.0,
        "max_xy_step": 0.002,
        "max_z_step": 0.001,
        "max_finger_delta": 0.00025,
        "force_close": True,
        "freeze_rotation": True,
    },
    "overview_cam_eye": [1.5, 0.1, 1.0],
    "overview_cam_target": [0.3, 0.1, 0.3],
    "render_camera": False,
    "render_cam_eye": [2.3, -1.9, 1.9],
    "render_cam_target": [0.3, 0.0, 0.35],
    "render_image_size": [720, 1280],
    "rewards": {},
    "reward_set": "simple_v3b",
    "reward_weights": {},
    "food_params": {},
    # Continuous demo (pipeline/0_state_teacher/demo.py only): None = the training scene; a mapping with any of
    # {bowls, food_pool, spacing} builds the demo scene (see FoodCellEnvCfg.demo).
    "demo": None,
}

DEMO_KEYS = {"bowls": "demo_bowls", "food_pool": "demo_food_pool", "spacing": "demo_spacing"}


def _resolve(registry: dict[str, str], name: str, kind: str):
    if name not in registry:
        raise KeyError(f"Unknown {kind} {name!r}. Available: {sorted(registry)}")
    module, attr = registry[name].split(":")
    return getattr(importlib.import_module(module), attr)


def _tuples(d: Mapping) -> dict:
    return {k: tuple(v) if isinstance(v, (list, tuple)) else v for k, v in d.items()}


def _geometry_kwargs(belt_kwargs: dict) -> dict:
    """Turn nested ``belt.bowl`` / ``belt.pallet`` dicts into geometry cfg instances."""
    from pickplace.assets.usd_builders import BowlGeometry, PalletGeometry

    out = dict(belt_kwargs)
    if "bowl" in out and isinstance(out["bowl"], Mapping):
        out["bowl"] = BowlGeometry(**_tuples(out["bowl"]))
    if "pallet" in out and isinstance(out["pallet"], Mapping):
        out["pallet"] = PalletGeometry(**_tuples(out["pallet"]))
    return out


def build_cell_env_cfg(env_cfg: Mapping):
    """Merge ``env_cfg`` over ``DEFAULT_ENV`` and construct the Isaac Lab env config."""
    # validated before the simulator imports below, so a typo fails fast (and is unit-testable without Isaac Sim)
    unknown = set(env_cfg) - set(DEFAULT_ENV)
    if unknown:
        raise ValueError(f"Unknown env config keys {sorted(unknown)}. Allowed: {sorted(DEFAULT_ENV)}")

    # Validated here rather than in make_env because action_adapter also decides stiff_arm below, i.e. it
    # changes the robot asset: a script that builds a cfg directly must not get the stiff arm with no
    # adapter and no error.
    merged = {**DEFAULT_ENV, **env_cfg}
    initial_joint_pos = merged["robot_initial_joint_pos"]
    if initial_joint_pos is not None:
        if not isinstance(initial_joint_pos, Mapping) or not initial_joint_pos:
            raise ValueError("robot_initial_joint_pos must be null or a non-empty joint-name mapping")
        if any(not isinstance(name, str) or not name for name in initial_joint_pos):
            raise ValueError("robot_initial_joint_pos keys must be non-empty joint names")
    adapter, action_mode = merged["action_adapter"], merged["action_mode"]
    if adapter not in ADAPTERS:
        raise ValueError(f"Unknown action adapter {adapter!r}. Available: {list(ADAPTERS)}")
    if adapter != "native" and action_mode != "joint_pos":
        raise ValueError(
            f"action_adapter={adapter!r} converts into the env's joint_pos action; set "
            f"action_mode='joint_pos' (got {action_mode!r})."
        )
    if merged["gripper_action_mode"] not in ("binary", "continuous"):
        raise ValueError("gripper_action_mode must be 'binary' or 'continuous'")
    if merged["ik_action_scale"] is not None and float(merged["ik_action_scale"]) <= 0:
        raise ValueError("ik_action_scale must be null or > 0")
    gate = {**DEFAULT_ENV["grasp_hold_gate"], **(merged["grasp_hold_gate"] or {})}
    unknown_gate = set(gate) - set(DEFAULT_ENV["grasp_hold_gate"])
    if unknown_gate:
        raise ValueError(
            f"Unknown grasp_hold_gate keys {sorted(unknown_gate)}. Allowed: {sorted(DEFAULT_ENV['grasp_hold_gate'])}"
        )
    if (
        int(gate["hold_steps"]) < 0
        or float(gate["max_upward_dz"]) < 0
        or float(gate["follow_gain"]) < 0
        or float(gate["max_follow_speed"]) <= 0
    ):
        raise ValueError(
            "grasp_hold_gate hold_steps/max_upward_dz/follow_gain must be >= 0 and max_follow_speed must be > 0"
        )
    if int(gate["hold_steps"]) > 0:
        if action_mode != "ee_delta_pose" or adapter != "native":
            raise ValueError("grasp_hold_gate requires native ee_delta_pose actions")
        if not merged["privileged_information"]:
            raise ValueError("grasp_hold_gate requires privileged_information=true for is_grasped")
        if not merged["grasp_observation"]:
            raise ValueError("grasp_hold_gate requires grasp_observation=true")
    option = {**DEFAULT_ENV["close_retract_option"], **(merged["close_retract_option"] or {})}
    unknown_option = set(option) - set(DEFAULT_ENV["close_retract_option"])
    if unknown_option:
        raise ValueError(
            f"Unknown close_retract_option keys {sorted(unknown_option)}. "
            f"Allowed: {sorted(DEFAULT_ENV['close_retract_option'])}"
        )
    if int(option["steps"]) < 0:
        raise ValueError("close_retract_option.steps must be >= 0")
    if int(option["steps"]) > 0:
        if action_mode != "ee_delta_pose" or adapter != "native":
            raise ValueError("close_retract_option requires native ee_delta_pose actions")
        if not merged["privileged_information"]:
            raise ValueError("close_retract_option requires object pose/velocity observations")
        if (
            float(option["retract_height"]) <= 0
            or float(option["trigger_distance"]) <= 0
            or float(option["follow_gain"]) < 0
            or float(option["max_follow_speed"]) <= 0
            or not 0.0 <= float(option["xy_blend"]) <= 1.0
        ):
            raise ValueError("invalid close_retract_option height/distance/gain/speed/xy_blend")
    pose_option = {
        **DEFAULT_ENV["grasp_pose_close_retract_option"],
        **(merged["grasp_pose_close_retract_option"] or {}),
    }
    unknown_pose_option = set(pose_option) - set(DEFAULT_ENV["grasp_pose_close_retract_option"])
    if unknown_pose_option:
        raise ValueError(
            f"Unknown grasp_pose_close_retract_option keys {sorted(unknown_pose_option)}. "
            f"Allowed: {sorted(DEFAULT_ENV['grasp_pose_close_retract_option'])}"
        )
    if bool(pose_option["enabled"]):
        if action_mode != "ee_delta_pose" or adapter != "native":
            raise ValueError("grasp_pose_close_retract_option requires native ee_delta_pose actions")
        if not merged["grasp_point_observation"]:
            raise ValueError("grasp_pose_close_retract_option requires grasp_point_observation=true")
        positive_pose_option = (
            "translation_threshold", "rotation_threshold", "retract_height", "position_gain",
            "rotation_gain", "max_position_step", "max_rotation_step",
        )
        if any(float(pose_option[key]) <= 0 for key in positive_pose_option):
            raise ValueError(f"grasp_pose_close_retract_option {positive_pose_option} must be > 0")
        if any(int(pose_option[key]) < 1 for key in ("approach_steps", "close_steps", "retract_steps")):
            raise ValueError("grasp_pose_close_retract_option approach/close/retract steps must be >= 1")
        if not -1.0 <= float(pose_option["close_action"]) < 1.0:
            raise ValueError("grasp_pose_close_retract_option.close_action must be in [-1, 1)")
    sphere_option = {
        **DEFAULT_ENV["stable_sphere_grasp_option"],
        **(merged["stable_sphere_grasp_option"] or {}),
    }
    unknown_sphere_option = set(sphere_option) - set(DEFAULT_ENV["stable_sphere_grasp_option"])
    if unknown_sphere_option:
        raise ValueError(
            f"Unknown stable_sphere_grasp_option keys {sorted(unknown_sphere_option)}. "
            f"Allowed: {sorted(DEFAULT_ENV['stable_sphere_grasp_option'])}"
        )
    if bool(sphere_option["enabled"]):
        if action_mode != "ee_delta_pose" or adapter != "native":
            raise ValueError("stable_sphere_grasp_option requires native ee_delta_pose actions")
        if not merged["grasp_point_observation"]:
            raise ValueError("stable_sphere_grasp_option requires grasp_point_observation=true")
        positive_sphere_option = tuple(
            key for key in DEFAULT_ENV["stable_sphere_grasp_option"]
            if key not in {"enabled", "alignment_steps", "close_steps", "hold_steps", "lift_steps", "lift_timeout_steps", "close_action"}
        )
        if any(float(sphere_option[key]) <= 0 for key in positive_sphere_option):
            raise ValueError(f"stable_sphere_grasp_option {positive_sphere_option} must be > 0")
        phase_keys = ("alignment_steps", "close_steps", "hold_steps", "lift_steps", "lift_timeout_steps")
        if any(int(sphere_option[key]) < 1 for key in phase_keys):
            raise ValueError("stable_sphere_grasp_option phase lengths must be >= 1")
        if int(sphere_option["lift_timeout_steps"]) < int(sphere_option["lift_steps"]):
            raise ValueError("stable_sphere_grasp_option lift timeout must cover lift steps")
        if not -1.0 <= float(sphere_option["close_action"]) < 1.0:
            raise ValueError("stable_sphere_grasp_option.close_action must be in [-1, 1)")
    residual_skill = {
        **DEFAULT_ENV["stable_sphere_residual_skill"],
        **(merged["stable_sphere_residual_skill"] or {}),
    }
    unknown_residual_skill = set(residual_skill) - set(DEFAULT_ENV["stable_sphere_residual_skill"])
    if unknown_residual_skill:
        raise ValueError(
            f"Unknown stable_sphere_residual_skill keys {sorted(unknown_residual_skill)}. "
            f"Allowed: {sorted(DEFAULT_ENV['stable_sphere_residual_skill'])}"
        )
    if bool(residual_skill["enabled"]):
        if bool(sphere_option["enabled"]):
            raise ValueError("enable only one stable sphere controller")
        if action_mode != "ee_delta_pose" or adapter != "native":
            raise ValueError("stable_sphere_residual_skill requires native ee_delta_pose actions")
        if not merged["grasp_point_observation"]:
            raise ValueError("stable_sphere_residual_skill requires grasp_point_observation=true")
        if merged["gripper_action_mode"] != "continuous":
            raise ValueError("stable_sphere_residual_skill requires continuous gripper control")
        phase_keys = ("alignment_steps", "close_steps", "hold_steps", "lift_steps", "lift_timeout_steps", "stable_steps")
        if any(int(residual_skill[key]) < 1 for key in phase_keys):
            raise ValueError("stable_sphere_residual_skill phase/stability steps must be >= 1")
        if int(residual_skill["lift_timeout_steps"]) < int(residual_skill["lift_steps"]):
            raise ValueError("stable_sphere_residual_skill lift timeout must cover lift steps")
        if residual_skill["pre_descent_height"] is not None:
            if not 0.0 < float(residual_skill["intercept_align_lead_distance"]) < float(
                residual_skill["intercept_lead_distance"]
            ):
                raise ValueError("residual skill align lead must be positive and below intercept lead")
            if not 0.0 < float(residual_skill["pre_descent_height"]) < float(
                residual_skill["hover_height"]
            ):
                raise ValueError("residual skill pre-descent height must be below hover height")
        if residual_skill["lift_target_slack"] is not None and float(
            residual_skill["lift_target_slack"]
        ) <= 0.0:
            raise ValueError("residual skill lift_target_slack must be null or positive")
        if residual_skill["lift_detach_height"] is not None and not (
            0.0
            < float(residual_skill["lift_detach_height"])
            < float(residual_skill["lift_height"])
        ):
            raise ValueError(
                "residual skill lift_detach_height must be null or between zero and lift_height"
            )
        for optional_positive in (
            "lift_xy_gain",
            "max_lift_xy_step",
            "lift_relative_z_gain",
        ):
            if residual_skill[optional_positive] is not None and float(
                residual_skill[optional_positive]
            ) <= 0.0:
                raise ValueError(f"residual skill {optional_positive} must be null or positive")
        if float(residual_skill["target_xy_feedforward_gain"]) < 0.0:
            raise ValueError("residual skill target_xy_feedforward_gain must be nonnegative")
        if residual_skill["max_target_xy_feedforward_step"] is not None and float(
            residual_skill["max_target_xy_feedforward_step"]
        ) <= 0.0:
            raise ValueError(
                "residual skill max_target_xy_feedforward_step must be null or positive"
            )
        if residual_skill["max_close_hold_xy_step"] is not None and float(
            residual_skill["max_close_hold_xy_step"]
        ) <= 0.0:
            raise ValueError("residual skill max_close_hold_xy_step must be null or positive")
        if residual_skill["lift_feedforward_decay_steps"] is not None and int(
            residual_skill["lift_feedforward_decay_steps"]
        ) < 1:
            raise ValueError("residual skill lift_feedforward_decay_steps must be null or >= 1")
        if residual_skill["lift_geometry_relatch_height"] is not None and not (
            0.0
            < float(residual_skill["lift_geometry_relatch_height"])
            < float(residual_skill["lift_height"])
        ):
            raise ValueError(
                "residual skill lift_geometry_relatch_height must be null or between zero and lift_height"
            )
        nonnegative = ("translation_residual_limit", "rotation_residual_limit")
        if any(not 0.0 <= float(residual_skill[key]) <= 1.0 for key in nonnegative):
            raise ValueError("stable_sphere_residual_skill residual limits must be in [0, 1]")
        if not -1.0 <= float(residual_skill["close_action"]) < 1.0:
            raise ValueError("stable_sphere_residual_skill.close_action must be in [-1, 1)")
    pick_place_skill = {
        **DEFAULT_ENV["stable_sphere_pick_place_skill"],
        **(merged["stable_sphere_pick_place_skill"] or {}),
    }
    unknown_pick_place = set(pick_place_skill) - set(DEFAULT_ENV["stable_sphere_pick_place_skill"])
    if unknown_pick_place:
        raise ValueError(
            f"Unknown stable_sphere_pick_place_skill keys {sorted(unknown_pick_place)}. "
            f"Allowed: {sorted(DEFAULT_ENV['stable_sphere_pick_place_skill'])}"
        )
    if bool(pick_place_skill["enabled"]):
        if not bool(residual_skill["enabled"]):
            raise ValueError("stable_sphere_pick_place_skill requires stable_sphere_residual_skill.enabled=true")
        if merged["task"] not in (
            "FoodRobot-MovingSphereStaticBowl-v0",
            "FoodRobot-StaticSphereMovingBowl-v0",
        ):
            raise ValueError(
                "stable_sphere_pick_place_skill requires a full sphere pick/place task"
            )
        positive = (
            "clear_height",
            "bowl_hover_height",
            "release_height",
            "position_gain",
            "max_position_step",
            "max_translation_action_delta",
            "max_lower_xy_step",
            "position_deadband",
            "clear_tolerance",
            "hover_tolerance",
            "release_tolerance",
            "retreat_height",
            "retreat_tolerance",
            "home_tolerance",
            "max_home_position_step",
        )
        if any(float(pick_place_skill[key]) <= 0.0 for key in positive):
            raise ValueError(f"stable_sphere_pick_place_skill {positive} must be > 0")
        if float(pick_place_skill["release_height"]) >= float(pick_place_skill["bowl_hover_height"]):
            raise ValueError("stable_sphere_pick_place_skill release height must be below hover height")
        if min(int(pick_place_skill["stable_steps"]), int(pick_place_skill["release_steps"])) < 1:
            raise ValueError("stable_sphere_pick_place_skill step counts must be >= 1")
        if pick_place_skill["home_position"] is not None and len(
            pick_place_skill["home_position"]
        ) != 3:
            raise ValueError("stable_sphere_pick_place_skill.home_position must contain three values")
        if not (
            -1.0
            <= float(pick_place_skill["close_action"])
            < float(pick_place_skill["open_action"])
            <= 1.0
        ):
            raise ValueError("stable_sphere_pick_place_skill gripper actions must satisfy -1 <= close < open <= 1")
    micro_lift = {
        **DEFAULT_ENV["geometric_micro_lift_option"],
        **(merged["geometric_micro_lift_option"] or {}),
    }
    unknown_micro_lift = set(micro_lift) - set(DEFAULT_ENV["geometric_micro_lift_option"])
    if unknown_micro_lift:
        raise ValueError(
            f"Unknown geometric_micro_lift_option keys {sorted(unknown_micro_lift)}. "
            f"Allowed: {sorted(DEFAULT_ENV['geometric_micro_lift_option'])}"
        )
    if bool(micro_lift["enabled"]):
        if action_mode != "ee_delta_pose" or adapter != "native":
            raise ValueError("geometric_micro_lift_option requires native ee_delta_pose actions")
        if int(merged["position_history_steps"]) < 1:
            raise ValueError("geometric_micro_lift_option requires position_history_steps >= 1")
        if int(micro_lift["close_steps"]) < 1 or int(micro_lift["lift_steps"]) < 1:
            raise ValueError("geometric_micro_lift_option close_steps/lift_steps must be >= 1")
        if int(micro_lift["hold_steps"]) < 0:
            raise ValueError("geometric_micro_lift_option hold_steps must be >= 0")
        positive = ("lift_height", "distance_threshold", "z_gain", "max_xy_step", "max_z_step")
        if (
            any(float(micro_lift[key]) <= 0 for key in positive)
            or float(micro_lift["xy_gain"]) < 0
            or float(micro_lift["max_finger_delta"]) < 0
        ):
            raise ValueError("geometric_micro_lift_option gains/steps/distances must be positive")
        minimum = float(micro_lift["min_close_fraction"])
        maximum = float(micro_lift["max_close_fraction"])
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("geometric_micro_lift_option close fractions must satisfy 0 <= min < max <= 1")

    from pickplace.belt import BeltCfg
    from pickplace.envs.cell_env_cfg import scale_physx_buffers
    from pickplace.rewards import DENSE_TERMS, isaac_weight, resolve_reward_weights

    c = merged
    if c["task"] not in TASK_CONFIGS:
        raise KeyError(f"Unknown task {c['task']!r}. Available: {sorted(TASK_CONFIGS)}")
    weights = resolve_reward_weights(c)
    arm = _resolve(ARMS, c["arm"], "arm")
    if c["ik_action_scale"] is not None:
        arm = copy.deepcopy(arm)
        arm.ik_action_scale = float(c["ik_action_scale"])
    food_cls = _resolve(FOODS, c["food"], "food")
    food = food_cls(**_tuples(c["food_params"]))

    demo = {}
    if c["demo"] is not None:
        unknown_demo = set(c["demo"]) - set(DEMO_KEYS)
        if unknown_demo:
            raise ValueError(f"Unknown demo keys {sorted(unknown_demo)}. Allowed: {sorted(DEMO_KEYS)}")
        demo = {"demo": True, **{DEMO_KEYS[k]: v for k, v in c["demo"].items() if v is not None}}

    env_cls = _resolve(TASK_CONFIGS, c["task"], "task")
    task_kwargs = {}
    if c["task"] in MOVING_STATIC_BOWL_TASKS | MOVING_LIFT_TASKS:
        task_kwargs = {
            "destination_bowl_pos": tuple(c["destination_bowl_pos"]),
            "destination_bowl_x_range": tuple(c["destination_bowl_x_range"]),
            "destination_bowl_y_range": tuple(c["destination_bowl_y_range"]),
        }
    if c["task"] in MOVING_LIFT_TASKS:
        task_kwargs["target_lift"] = float(c["target_lift"])
        if float(c["geometric_slip_penalty"]) < 0:
            raise ValueError("geometric_slip_penalty must be >= 0")
        if int(c["geometric_success_steps"]) < 1:
            raise ValueError("geometric_success_steps must be >= 1")
        close_curriculum = {
            **DEFAULT_ENV["geometric_close_curriculum"],
            **(c["geometric_close_curriculum"] or {}),
        }
        unknown_close = set(close_curriculum) - set(DEFAULT_ENV["geometric_close_curriculum"])
        if unknown_close:
            raise ValueError(
                f"Unknown geometric_close_curriculum keys {sorted(unknown_close)}. "
                f"Allowed: {sorted(DEFAULT_ENV['geometric_close_curriculum'])}"
            )
        if int(close_curriculum["success_steps"]) < 0:
            raise ValueError("geometric_close_curriculum.success_steps must be >= 0")
        if float(close_curriculum["distance_threshold"]) <= 0:
            raise ValueError("geometric_close_curriculum.distance_threshold must be > 0")
        minimum = float(close_curriculum["min_close_fraction"])
        maximum = float(close_curriculum["max_close_fraction"])
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("geometric_close_curriculum close fractions must satisfy 0 <= min < max <= 1")
        if float(close_curriculum["far_close_penalty"]) < 0:
            raise ValueError("geometric_close_curriculum.far_close_penalty must be >= 0")
        close_lift_bridge = {
            **DEFAULT_ENV["geometric_close_lift_bridge"],
            **(c["geometric_close_lift_bridge"] or {}),
        }
        unknown_bridge = set(close_lift_bridge) - set(DEFAULT_ENV["geometric_close_lift_bridge"])
        if unknown_bridge:
            raise ValueError(
                f"Unknown geometric_close_lift_bridge keys {sorted(unknown_bridge)}. "
                f"Allowed: {sorted(DEFAULT_ENV['geometric_close_lift_bridge'])}"
            )
        if int(close_lift_bridge["close_steps"]) < 1:
            raise ValueError("geometric_close_lift_bridge.close_steps must be >= 1")
        if int(close_lift_bridge["success_steps"]) < 1:
            raise ValueError("geometric_close_lift_bridge.success_steps must be >= 1")
        positive_bridge = (
            "distance_threshold", "prelift_max_height", "target_lift", "max_relative_drift"
        )
        if any(float(close_lift_bridge[key]) <= 0 for key in positive_bridge):
            raise ValueError(f"geometric_close_lift_bridge {positive_bridge} must be > 0")
        progress_weights = ("tcp_progress_weight", "object_progress_weight")
        if any(float(close_lift_bridge[key]) < 0 for key in progress_weights):
            raise ValueError(f"geometric_close_lift_bridge {progress_weights} must be >= 0")
        if sum(float(close_lift_bridge[key]) for key in progress_weights) <= 0:
            raise ValueError("geometric_close_lift_bridge progress weights cannot both be zero")
        bridge_minimum = float(close_lift_bridge["min_close_fraction"])
        bridge_maximum = float(close_lift_bridge["max_close_fraction"])
        if not 0.0 <= bridge_minimum < bridge_maximum <= 1.0:
            raise ValueError(
                "geometric_close_lift_bridge close fractions must satisfy 0 <= min < max <= 1"
            )
        geometric_stability = {
            **DEFAULT_ENV["geometric_grasp_stability"],
            **(c["geometric_grasp_stability"] or {}),
        }
        unknown_geometric_stability = (
            set(geometric_stability) - set(DEFAULT_ENV["geometric_grasp_stability"])
        )
        if unknown_geometric_stability:
            raise ValueError(
                f"Unknown geometric_grasp_stability keys {sorted(unknown_geometric_stability)}. "
                f"Allowed: {sorted(DEFAULT_ENV['geometric_grasp_stability'])}"
            )
        if int(geometric_stability["stable_steps"]) < 0:
            raise ValueError("geometric_grasp_stability.stable_steps must be >= 0")
        positive_geometric = (
            "distance_threshold", "relative_speed_threshold", "distance_std", "relative_speed_std"
        )
        if any(float(geometric_stability[key]) <= 0 for key in positive_geometric):
            raise ValueError(f"geometric_grasp_stability {positive_geometric} must be > 0")
        nonnegative_geometric = (
            "hold_reward", "premature_close_penalty", "confidence_loss_penalty"
        )
        if any(float(geometric_stability[key]) < 0 for key in nonnegative_geometric):
            raise ValueError(f"geometric_grasp_stability {nonnegative_geometric} must be >= 0")
        lift_velocity = {
            **DEFAULT_ENV["geometric_lift_velocity"],
            **(c["geometric_lift_velocity"] or {}),
        }
        unknown_lift_velocity = set(lift_velocity) - set(DEFAULT_ENV["geometric_lift_velocity"])
        if unknown_lift_velocity:
            raise ValueError(
                f"Unknown geometric_lift_velocity keys {sorted(unknown_lift_velocity)}. "
                f"Allowed: {sorted(DEFAULT_ENV['geometric_lift_velocity'])}"
            )
        if not 0.0 <= float(lift_velocity["weight"]) <= 1.0:
            raise ValueError("geometric_lift_velocity.weight must be in [0, 1]")
        if float(lift_velocity["std"]) <= 0:
            raise ValueError("geometric_lift_velocity.std must be > 0")
        task_kwargs.update(
            geometry_only_grasp=bool(c["geometry_only_grasp"]),
            geometric_slip_penalty=float(c["geometric_slip_penalty"]),
            geometric_success_steps=int(c["geometric_success_steps"]),
            geometric_close_success_steps=int(close_curriculum["success_steps"]),
            geometric_close_distance=float(close_curriculum["distance_threshold"]),
            geometric_close_min_fraction=minimum,
            geometric_close_max_fraction=maximum,
            geometric_far_close_penalty=float(close_curriculum["far_close_penalty"]),
            geometric_bridge_enabled=bool(close_lift_bridge["enabled"]),
            geometric_bridge_progress_reward=bool(close_lift_bridge["progress_reward"]),
            geometric_bridge_close_steps=int(close_lift_bridge["close_steps"]),
            geometric_bridge_distance=float(close_lift_bridge["distance_threshold"]),
            geometric_bridge_min_fraction=bridge_minimum,
            geometric_bridge_max_fraction=bridge_maximum,
            geometric_bridge_prelift_max_height=float(close_lift_bridge["prelift_max_height"]),
            geometric_bridge_target_lift=float(close_lift_bridge["target_lift"]),
            geometric_bridge_max_relative_drift=float(close_lift_bridge["max_relative_drift"]),
            geometric_bridge_success_steps=int(close_lift_bridge["success_steps"]),
            geometric_bridge_tcp_progress_weight=float(close_lift_bridge["tcp_progress_weight"]),
            geometric_bridge_object_progress_weight=float(close_lift_bridge["object_progress_weight"]),
            geometric_stable_steps=int(geometric_stability["stable_steps"]),
            geometric_stable_distance=float(geometric_stability["distance_threshold"]),
            geometric_stable_relative_speed=float(geometric_stability["relative_speed_threshold"]),
            geometric_stable_distance_std=float(geometric_stability["distance_std"]),
            geometric_stable_relative_speed_std=float(geometric_stability["relative_speed_std"]),
            geometric_stable_hold_reward=float(geometric_stability["hold_reward"]),
            geometric_premature_close_penalty=float(geometric_stability["premature_close_penalty"]),
            geometric_confidence_loss_penalty=float(geometric_stability["confidence_loss_penalty"]),
            geometric_lift_velocity_weight=float(lift_velocity["weight"]),
            geometric_lift_velocity_std=float(lift_velocity["std"]),
        )
        stability = {**DEFAULT_ENV["grasp_stability"], **(c["grasp_stability"] or {})}
        unknown_stability = set(stability) - set(DEFAULT_ENV["grasp_stability"])
        if unknown_stability:
            raise ValueError(
                f"Unknown grasp_stability keys {sorted(unknown_stability)}. "
                f"Allowed: {sorted(DEFAULT_ENV['grasp_stability'])}"
            )
        if int(stability["hold_steps"]) < 0:
            raise ValueError("grasp_stability.hold_steps must be >= 0")
        if float(stability["hold_reward"]) < 0 or float(stability["slip_penalty"]) < 0:
            raise ValueError("grasp_stability hold_reward/slip_penalty must be >= 0")
        if float(stability["premature_lift_penalty"]) < 0:
            raise ValueError("grasp_stability.premature_lift_penalty must be >= 0")
        prelift_max_height = stability["prelift_max_height"]
        if prelift_max_height is not None and float(prelift_max_height) <= 0:
            raise ValueError("grasp_stability.prelift_max_height must be null or > 0")
        if (prelift_max_height is not None or stability["require_stable_success"]) and int(stability["hold_steps"]) < 1:
            raise ValueError("pre-lift staging requires grasp_stability.hold_steps >= 1")
        task_kwargs.update(
            stable_grasp_steps=int(stability["hold_steps"]),
            stable_grasp_reward=float(stability["hold_reward"]),
            grasp_slip_penalty=float(stability["slip_penalty"]),
            stable_prelift_max_height=(None if prelift_max_height is None else float(prelift_max_height)),
            premature_lift_penalty=float(stability["premature_lift_penalty"]),
            require_stable_grasp_success=bool(stability["require_stable_success"]),
        )
    cfg = env_cls(
        arm=arm,
        food=food,
        belt=BeltCfg(**_geometry_kwargs(_tuples(c["belt"]))),
        action_mode=c["action_mode"],
        gripper_action_mode=c["gripper_action_mode"],
        # an adapter commands joint targets; the plain gains track those poorly (FoodCellEnvCfg.stiff_arm)
        stiff_arm=c["action_adapter"] != "native",
        cameras=bool(c["cameras"]),
        camera_depth=bool(c["camera_depth"]),
        image_size=tuple(c["image_size"]),
        frame_stack=int(c["frame_stack"]),
        privileged_information=bool(c["privileged_information"]),
        grasp_observation=bool(c["grasp_observation"]),
        temporal_history_steps=int(c["temporal_history_steps"]),
        kinematic_grasp_proxy_observation=bool(c["kinematic_grasp_proxy_observation"]),
        position_history_steps=int(c["position_history_steps"]),
        grasp_point_observation=bool(c["grasp_point_observation"]),
        speed_free_observations=bool(c["speed_free_observations"]),
        success_bonus=weights["success"],
        bowl_failure_penalty=-weights["bowl_failure"],
        food_drop_penalty=-weights["food_dropped"],
        success_settle_steps=int(c["success_settle_steps"]),
        success_requires_home=bool(c["success_requires_home"]),
        home_tolerance=float(c["home_tolerance"]),
        robot_reset=_tuples({**DEFAULT_ENV["robot_reset"], **c["robot_reset"]}),
        ingredient_bowl_pos=tuple(c["ingredient_bowl_pos"]),
        ingredient_bowl_x_range=tuple(c["ingredient_bowl_x_range"]),
        ingredient_bowl_y_range=tuple(c["ingredient_bowl_y_range"]),
        overview_cam_eye=tuple(c["overview_cam_eye"]),
        overview_cam_target=tuple(c["overview_cam_target"]),
        render_camera=bool(c["render_camera"]),
        render_cam_eye=tuple(c["render_cam_eye"]),
        render_cam_target=tuple(c["render_cam_target"]),
        render_image_size=tuple(c["render_image_size"]),
        **task_kwargs,
        **demo,
    )
    if initial_joint_pos is not None:
        cfg.scene.robot.init_state.joint_pos.update(
            {name: float(value) for name, value in initial_joint_pos.items()}
        )
    cfg.scene.num_envs = int(c["num_envs"])
    scale_physx_buffers(cfg.sim.physics, cfg.scene.num_envs)
    cfg.seed = c["seed"]
    cfg.sim.device = c["device"]
    for term in DENSE_TERMS:
        # Isaac Lab keeps computing every dense term (non-zero weight) so the reward-term vector can recover
        # the unweighted value; the training reward itself comes from LineariseRewards in make_env.
        getattr(cfg.rewards, term).weight = isaac_weight(weights[term])
    return cfg
