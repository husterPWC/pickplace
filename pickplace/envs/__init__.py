"""Gym registration. Importing this module is cheap; the env config imports Isaac Lab lazily via gym.make."""

import gymnasium as gym

gym.register(
    id="FoodRobot-Cell-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "pickplace.envs.cell_env_cfg:FoodCellEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="FoodRobot-MovingCubeStaticBowl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "pickplace.envs.moving_cube_env_cfg:MovingCubeStaticBowlEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="FoodRobot-MovingCubeLift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "pickplace.envs.moving_cube_env_cfg:MovingCubeLiftEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="FoodRobot-MovingSphereStaticBowl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "pickplace.envs.moving_sphere_env_cfg:MovingSphereStaticBowlEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="FoodRobot-MovingSphereLift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "pickplace.envs.moving_sphere_env_cfg:MovingSphereLiftEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="FoodRobot-MovingSphereGapLift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "pickplace.envs.moving_sphere_env_cfg:MovingSphereGapLiftEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="FoodRobot-MovingSphereCanonicalGapLift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "pickplace.envs.moving_sphere_env_cfg:MovingSphereCanonicalGapLiftEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="FoodRobot-MovingSphereA0Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "pickplace.envs.moving_sphere_env_cfg:MovingSphereA0LiftEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="FoodRobot-MovingSphereResidualGapLift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "pickplace.envs.moving_sphere_env_cfg:MovingSphereResidualGapLiftEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="FoodRobot-StaticSphereMovingBowl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            "pickplace.envs.static_sphere_moving_bowl_env_cfg:"
            "StaticSphereMovingBowlEnvCfg"
        )
    },
    disable_env_checker=True,
)
