"""Build the formal moving-sphere task and verify reset and conveyor motion."""

from _common import finish

from pickplace.app import launch_app

app = launch_app(headless=True, enable_cameras=False)

import torch  # noqa: E402

from pickplace.torchrl_env import make_env  # noqa: E402


def main():
    cfg = {
        "task": "FoodRobot-MovingSphereStaticBowl-v0",
        "num_envs": 4,
        "arm": "franka",
        "food": "moving_sphere",
        "action_mode": "ee_delta_pose",
        "cameras": False,
        "privileged_information": True,
        "belt": {
            "speed": 0.10,
            "speed_noise": 0.0,
            "place_window": 4.0,
            "pallet_start_range": [0.0, 0.0],
            "bowl_offset_x": [0.0, 0.0],
            "bowl_offset_y": [0.0, 0.0],
        },
        "destination_bowl_pos": [0.45, -0.10, 0.0],
        "destination_bowl_x_range": [0.45, 0.45],
        "destination_bowl_y_range": [-0.10, -0.10],
        "reward_set": "moving_sphere_v1",
        "success_requires_home": True,
        "device": "cuda:0",
        "seed": 0,
    }
    env = make_env(cfg)
    td = env.reset()
    x0 = td["privileged", "food_pos"][:, 0].clone()
    bowl0 = td["belt", "bowl_pos"].clone()
    speed0 = td["belt", "speed"].clone()
    velocity0 = td["privileged", "food_lin_vel"].clone()

    for _ in range(10):
        td.set("action", torch.zeros(env.action_spec.shape, device=env.device))
        _, td = env.step_and_maybe_reset(td)

    x1 = td["privileged", "food_pos"][:, 0]
    bowl1 = td["belt", "bowl_pos"]
    finish(
        bool(
            torch.all(x1 > x0 + 0.005)
            and torch.allclose(bowl0, bowl1, atol=1e-5)
            and torch.allclose(speed0, torch.full_like(speed0, 0.10), atol=1e-6)
            and torch.allclose(velocity0[:, 0], torch.full_like(velocity0[:, 0], 0.10), atol=1e-4)
        ),
        task=env.base_env._env.unwrapped.spec.id,
        food_name=env.base_env._env.unwrapped.cfg.food.name,
        radius=env.base_env._env.unwrapped.cfg.food.item_radius,
        mass=env.base_env._env.unwrapped.cfg.food.mass,
        sphere_dx=(x1 - x0).tolist(),
        bowl_drift=torch.linalg.vector_norm(bowl1 - bowl0, dim=-1).tolist(),
        outcome_keys=sorted(td["outcome"].keys()),
    )


try:
    main()
except Exception as exc:
    import traceback

    traceback.print_exc()
    finish(False, error=repr(exc))
