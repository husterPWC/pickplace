import pytest

pytestmark = pytest.mark.isaac


def test_independent_moving_cube_task_reset_and_motion(run_scenario):
    result = run_scenario("moving_cube_task")
    assert result["action_dim"] == 7
    assert result["shapes"]["belt"] == {"bowl_pos": [3], "speed": [1]}
    assert result["shapes"]["privileged"] == {
        "food_pos": [3],
        "food_lin_vel": [3],
        "is_grasped": [1],
        "food_quat": [4],
    }
    assert set(result["outcome_keys"]) == {"success", "cube_exited_zone", "cube_off_belt", "time_out"}
    assert max(result["bowl_drift"]) < 1e-5
    assert result["exit_ignore_above_z"] > 0.1
    assert result["lateral_check_max_z"] > 0.1
    assert result["lateral_safe_radius"] > 0.05


def test_independent_moving_sphere_task_reset_and_motion(run_scenario):
    result = run_scenario("moving_sphere_task")
    assert result["task"] == "FoodRobot-MovingSphereStaticBowl-v0"
    assert result["food_name"] == "conveyor_sphere"
    assert result["radius"] == pytest.approx(0.02)
    assert result["mass"] == pytest.approx(0.03)
    assert min(result["sphere_dx"]) > 0.005
    assert max(result["bowl_drift"]) < 1e-5
    assert set(result["outcome_keys"]) == {"success", "cube_exited_zone", "cube_off_belt", "time_out"}
