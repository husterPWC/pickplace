"""Simulator-free contract for G2 RGB-D teacher-data collection."""

from __future__ import annotations

from collections.abc import Mapping


RGB_KEYS = (("pixels", "overview_rgb"), ("pixels", "wrist_rgb"))
DEPTH_KEYS = (("pixels", "overview_depth"), ("pixels", "wrist_depth"))
DEPLOYABLE_PROPRIO_KEYS = (
    ("proprio", "joint_pos_rel"),
    ("proprio", "gripper_pos"),
    ("proprio", "ee_pos"),
    ("proprio", "ee_quat"),
    ("proprio", "last_action"),
)
CALIBRATED_SCENE_KEYS = (("belt", "bowl_pos"),)
TEACHER_ONLY_GROUPS = ("grasp_guidance", "position_history")
FORBIDDEN_STUDENT_SIGNALS = (
    "object_velocity",
    "belt_speed",
    "contact",
    "force",
    "tactile",
    "motor_current",
    "torque",
)

JAKA_VISUAL_TASKS = {
    "moving_sphere_static_bowl": {
        "effective_task": "FoodRobot-MovingSphereStaticBowl-v0",
        "actor_source_tasks": (
            "FoodRobot-MovingSphereResidualGapLift-v0",
            "FoodRobot-MovingSphereStaticBowl-v0",
        ),
        "success_requires_home": False,
        "description": "moving sphere at 0.06--0.08 m/s to a static bowl",
    },
    "static_sphere_moving_bowl": {
        "effective_task": "FoodRobot-StaticSphereMovingBowl-v0",
        "actor_source_tasks": ("FoodRobot-StaticSphereMovingBowl-v0",),
        "success_requires_home": True,
        "description": "static sphere to a moving bowl at 0.10 m/s",
    },
}

PROFILES = {
    "smoke": {"frames": 4_096, "num_envs": 32, "rollout_steps": 8},
    "pilot": {"frames": 100_000, "num_envs": 128, "rollout_steps": 8},
    "full": {"frames": 1_000_000, "num_envs": 256, "rollout_steps": 8},
}


def profile(name: str) -> dict[str, int]:
    """Return a copy so callers cannot mutate the global contract."""
    if name not in PROFILES:
        raise ValueError(f"unknown G2 collection profile {name!r}; choose {sorted(PROFILES)}")
    return dict(PROFILES[name])


def validate_g1_teacher_payload(payload: Mapping) -> dict[str, object]:
    """Reject a checkpoint that cannot serve as the frozen G2 teacher."""
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint has no config mapping")
    env = config.get("env")
    network = config.get("network")
    if not isinstance(env, Mapping) or not isinstance(network, Mapping):
        raise ValueError("checkpoint config must contain env and network mappings")
    if env.get("task") != "FoodRobot-MovingSphereStaticBowl-v0":
        raise ValueError("G2 requires the moving-sphere/static-bowl G1 task")
    if not bool(env.get("speed_free_observations")):
        raise ValueError("G2 teacher must be the speed-free policy")
    if not bool((env.get("stable_sphere_pick_place_skill") or {}).get("enabled")):
        raise ValueError("G2 teacher must include the complete G1 pick/place controller")
    actor_keys = [tuple(key) for key in network.get("actor_in_keys", ())]
    flat = {leaf for key in actor_keys for leaf in key}
    forbidden = sorted(signal for signal in FORBIDDEN_STUDENT_SIGNALS if signal in flat)
    if forbidden:
        raise ValueError(f"teacher actor exposes forbidden signals: {forbidden}")
    return {
        "task": env["task"],
        "frames": int(payload.get("frames", 0)),
        "speed_free": True,
        "pick_place_controller": True,
        "actor_in_keys": [list(key) for key in actor_keys],
        "student_runtime_keys": [
            list(key)
            for key in (*RGB_KEYS, *DEPTH_KEYS, *DEPLOYABLE_PROPRIO_KEYS, *CALIBRATED_SCENE_KEYS)
        ],
        "calibrated_runtime_value": "belt/bowl_pos (static destination-bowl pose in the cell frame)",
        "teacher_only_groups": list(TEACHER_ONLY_GROUPS),
        "forbidden_student_signals": list(FORBIDDEN_STUDENT_SIGNALS),
        "action_semantics": "teacher_residual_before_environment_g1_controller",
    }


def validate_jaka_visual_teacher_payload(
    actor_payload: Mapping,
    env_payload: Mapping,
    task_profile: str,
) -> dict[str, object]:
    """Validate one frozen JAKA actor plus its effective full-task environment.

    The moving-sphere task deliberately uses the accepted J0-S1 actor weights
    with the later J1 full pick/place environment.  Keeping the two payloads
    explicit prevents a collector from silently reverting to the lift-only
    task stored in the actor checkpoint.
    """
    if task_profile not in JAKA_VISUAL_TASKS:
        raise ValueError(
            f"unknown JAKA visual task {task_profile!r}; choose {sorted(JAKA_VISUAL_TASKS)}"
        )
    spec = JAKA_VISUAL_TASKS[task_profile]
    actor_config = actor_payload.get("config")
    env_config = env_payload.get("config")
    if not isinstance(actor_config, Mapping) or not isinstance(env_config, Mapping):
        raise ValueError("actor and environment checkpoints must contain config mappings")
    actor_env = actor_config.get("env")
    effective_env = env_config.get("env")
    network = actor_config.get("network")
    if not all(isinstance(value, Mapping) for value in (actor_env, effective_env, network)):
        raise ValueError("checkpoint configs must contain env and network mappings")

    if actor_env.get("arm") != "zu5_ag95" or effective_env.get("arm") != "zu5_ag95":
        raise ValueError("JAKA visual collection requires arm=zu5_ag95 in both checkpoints")
    if actor_env.get("task") not in spec["actor_source_tasks"]:
        raise ValueError(
            f"actor source task {actor_env.get('task')!r} is invalid for {task_profile}"
        )
    if effective_env.get("task") != spec["effective_task"]:
        raise ValueError(
            f"effective task {effective_env.get('task')!r} does not match "
            f"{spec['effective_task']!r}"
        )
    if not bool(actor_env.get("speed_free_observations")):
        raise ValueError("JAKA actor must use speed-free observations")
    if not bool(effective_env.get("speed_free_observations")):
        raise ValueError("effective JAKA environment must keep speed-free observations")
    if not bool((effective_env.get("stable_sphere_pick_place_skill") or {}).get("enabled")):
        raise ValueError("effective JAKA environment must enable the complete pick/place skill")
    if bool(effective_env.get("success_requires_home", False)) != bool(spec["success_requires_home"]):
        raise ValueError("effective environment has the wrong Home success contract")

    actor_keys = [tuple(key) for key in network.get("actor_in_keys", ())]
    flat = {leaf for key in actor_keys for leaf in key}
    forbidden = sorted(signal for signal in FORBIDDEN_STUDENT_SIGNALS if signal in flat)
    if forbidden:
        raise ValueError(f"teacher actor exposes forbidden signals: {forbidden}")

    return {
        "task_profile": task_profile,
        "description": spec["description"],
        "robot": "jaka_zu5_ag95",
        "actor_source_task": actor_env["task"],
        "effective_task": effective_env["task"],
        "actor_frames": int(actor_payload.get("frames", 0)),
        "speed_free": True,
        "pick_place_controller": True,
        "success_requires_home": bool(spec["success_requires_home"]),
        "actor_in_keys": [list(key) for key in actor_keys],
        "student_runtime_keys": [
            list(key)
            for key in (*RGB_KEYS, *DEPTH_KEYS, *DEPLOYABLE_PROPRIO_KEYS, *CALIBRATED_SCENE_KEYS)
        ],
        "calibrated_runtime_value": (
            "belt/bowl_pos; fixed calibration for the static bowl and visual/"
            "calibrated tracking for the moving bowl"
        ),
        "teacher_only_groups": list(TEACHER_ONLY_GROUPS),
        "forbidden_student_signals": list(FORBIDDEN_STUDENT_SIGNALS),
        "action_semantics": "teacher_residual_before_environment_jaka_skill_controller",
    }


def estimated_shard_bytes(frames: int, image_size: int = 84) -> int:
    """Conservative RGB-D + teacher-label storage estimate.

    RGB is uint8, depth is float16 metres.  The fixed allowance covers G1
    grasp candidates, position history, proprioception, actions and outcomes.
    """
    if frames < 1 or image_size < 1:
        raise ValueError("frames and image_size must be positive")
    rgb = 2 * image_size * image_size * 3
    depth = 2 * image_size * image_size * 2
    teacher_and_metadata = 12_500
    return int(frames) * (rgb + depth + teacher_and_metadata)
