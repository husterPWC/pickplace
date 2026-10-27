"""Build and step the independent sphere_gap task."""

from _common import finish

from pickplace.app import launch_app

app = launch_app(headless=True, enable_cameras=False, device="cuda:0")

import torch  # noqa: E402

from pickplace.torchrl_env import make_env  # noqa: E402


def main():
    cfg = {
        "task": "FoodRobot-MovingSphereGapLift-v0",
        "num_envs": 4,
        "arm": "franka",
        "food": "moving_sphere",
        "action_mode": "ee_delta_pose",
        "action_adapter": "native",
        "gripper_action_mode": "continuous",
        "ik_action_scale": 0.02,
        "cameras": False,
        "privileged_information": False,
        "grasp_observation": False,
        "grasp_point_observation": True,
        "speed_free_observations": True,
        "belt": {
            "speed": 0.02,
            "speed_noise": 0.5,
            "place_window": 20.0,
            "pallet_start_range": [-0.04, 0.04],
            "bowl_offset_x": [-0.01, 0.01],
            "bowl_offset_y": [-0.02, 0.02],
        },
        "target_lift": 0.03,
        "reward_set": "sphere_gap_lift_v1",
        "grasp_pose_close_retract_option": {"enabled": True},
        "device": "cuda:0",
    }
    env = make_env(cfg)
    td = env.reset()
    points = td["grasp_guidance", "points_ee"]
    pose = td["grasp_guidance", "best_pose_delta"]
    forbidden = {
        ("belt", "speed"),
        ("proprio", "joint_vel_rel"),
        ("privileged", "food_lin_vel"),
        ("privileged", "is_grasped"),
    }
    keys = set(td.keys(True, True))
    cfg_built = env.base_env._env.unwrapped.cfg
    for _ in range(3):
        td.set("action", torch.zeros(env.action_spec.shape, device=env.device))
        transition, td = env.step_and_maybe_reset(td)
        assert torch.isfinite(transition["next", "reward"]).all()
    finish(
        bool(
            points.shape[1:] == (40, 20, 3)
            and pose.shape[1:] == (6,)
            and env.action_spec.shape[-1] == 7
            and forbidden.isdisjoint(keys)
            and torch.isfinite(points).all()
            and torch.isfinite(pose).all()
        ),
        action_dim=env.action_spec.shape[-1],
        point_shape=list(points.shape[1:]),
        pose_shape=list(pose.shape[1:]),
        ik_action_scale=cfg_built.arm.ik_action_scale,
        gripper_action_mode=cfg_built.gripper_action_mode,
        sensors=[name for name in ("left_finger_contact", "right_finger_contact") if name in env.base_env._env.unwrapped.scene.sensors],
        observation_keys=sorted("/".join(key) if isinstance(key, tuple) else str(key) for key in keys),
    )


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
