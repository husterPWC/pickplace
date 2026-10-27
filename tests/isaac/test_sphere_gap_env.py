import pytest

pytestmark = pytest.mark.isaac


def test_sphere_gap_contract(run_scenario):
    result = run_scenario("sphere_gap_env")
    assert result["action_dim"] == 7
    assert result["point_shape"] == [40, 20, 3]
    assert result["pose_shape"] == [6]
    assert result["ik_action_scale"] == pytest.approx(0.02)
    assert result["gripper_action_mode"] == "continuous"
    assert result["sensors"] == ["left_finger_contact", "right_finger_contact"]
