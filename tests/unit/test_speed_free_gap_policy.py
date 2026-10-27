import importlib.util
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "pipeline" / "0_state_teacher" / "config_moving_cube_c0p_gap_speed_free.yaml"
DIRECT_CONFIG = ROOT / "pipeline" / "0_state_teacher" / "config_moving_cube_c0t_gap_direct_rl.yaml"
CLOSE_CONFIG = ROOT / "pipeline" / "0_state_teacher" / "config_moving_cube_c0u_close_direct_rl.yaml"
LIFT_CONFIG = ROOT / "pipeline" / "0_state_teacher" / "config_moving_cube_c0v_lift_direct_rl.yaml"
BRIDGE_CONFIG = ROOT / "pipeline" / "0_state_teacher" / "config_moving_cube_c0w_micro_lift_bridge.yaml"
GATED_BRIDGE_CONFIG = ROOT / "pipeline" / "0_state_teacher" / "config_moving_cube_c0x_gated_micro_lift.yaml"
UTILS = ROOT / "pipeline" / "0_state_teacher" / "utils.py"
SPEC = importlib.util.spec_from_file_location("teacher_speed_free_test_utils", UTILS)
teacher_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(teacher_utils)


def test_c0p_actor_contract_contains_no_explicit_speed_or_contact_key():
    cfg = yaml.safe_load(CONFIG.read_text())
    env = cfg["env"]
    assert env["speed_free_observations"] is True
    assert env["privileged_information"] is False
    assert env["temporal_history_steps"] == 0
    assert env["position_history_steps"] == 10
    assert env["grasp_point_observation"] is True

    actor_keys = {tuple(key) for key in cfg["network"]["actor_in_keys"]}
    forbidden = {"vel", "speed", "motion", "grasped", "proxy", "contact", "force", "tactile", "current"}
    assert not any(any(token in part for token in forbidden) for key in actor_keys for part in key)
    assert ("grasp_guidance", "points_ee") in actor_keys
    assert ("position_history", "position_action") in actor_keys


def test_c0t_is_direct_rl_from_scratch_with_speed_only_randomized_in_environment():
    cfg = yaml.safe_load(DIRECT_CONFIG.read_text())
    env = cfg["env"]
    assert env["speed_free_observations"] is True
    assert env["privileged_information"] is False
    assert env["belt"]["speed"] == 0.10
    assert env["belt"]["speed_noise"] == 0.20
    assert env["geometric_lift_velocity"]["weight"] == 0.0
    assert cfg["init_checkpoint"] is None
    assert cfg["resume_checkpoint"] is None
    assert "distill" not in cfg

    actor_keys = {tuple(key) for key in cfg["network"]["actor_in_keys"]}
    critic_keys = {tuple(key) for key in cfg["network"]["critic_in_keys"]}
    forbidden = {"vel", "speed", "motion", "grasped", "proxy", "contact", "force", "tactile", "current"}
    assert actor_keys == critic_keys
    assert not any(any(token in part for token in forbidden) for key in actor_keys for part in key)
    assert ("grasp_guidance", "points_ee") in actor_keys
    assert ("position_history", "position_action") in actor_keys
    assert cfg["loss"]["freeze_actor_scale"] is False


def test_grasp_point_gru_accepts_batch_and_time_leading_dimensions():
    shapes = [(3,), (4,), (4, 16, 3), (10, 12)]
    model = teacher_utils.GraspPointGRUStateNet(
        shapes=shapes,
        grasp_index=2,
        history_index=3,
        hidden=[32, 16],
        point_hidden=8,
        temporal_hidden=7,
        out_dim=7,
        out_gain=0.01,
    )
    inputs = (
        torch.randn(5, 6, 3),
        torch.randn(5, 6, 4),
        torch.randn(5, 6, 4, 16, 3),
        torch.randn(5, 6, 10, 12),
    )
    output = model(*inputs)
    assert output.shape == (5, 6, 7)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_zero_output_gain_preserves_a_geometric_residual_baseline():
    model = teacher_utils.GraspPointGRUStateNet(
        shapes=[(3,), (4,), (4, 16, 3), (10, 12)],
        grasp_index=2,
        history_index=3,
        hidden=[32, 16],
        point_hidden=8,
        temporal_hidden=7,
        out_dim=7,
        out_gain=0.0,
    )
    output = model(
        torch.randn(5, 3),
        torch.randn(5, 4),
        torch.randn(5, 4, 16, 3),
        torch.randn(5, 10, 12),
    )
    assert torch.equal(output, torch.zeros_like(output))


def test_c0u_and_c0v_form_a_speed_randomized_close_then_lift_curriculum():
    close = yaml.safe_load(CLOSE_CONFIG.read_text())
    lift = yaml.safe_load(LIFT_CONFIG.read_text())

    assert close["env"]["reward_set"] == "moving_cube_close_v1"
    assert close["env"]["belt"] == {"speed": 0.06, "speed_noise": 0.25}
    assert close["env"]["geometric_close_curriculum"] == {
        "success_steps": 5,
        "distance_threshold": 0.045,
        "min_close_fraction": 0.15,
        "max_close_fraction": 0.90,
        "far_close_penalty": 0.50,
    }
    assert close["init_checkpoint"] is None

    assert lift["env"]["belt"] == {"speed": 0.08, "speed_noise": 0.25}
    assert lift["env"]["geometric_close_curriculum"]["success_steps"] == 0
    assert lift["init_checkpoint"] is None  # supplied explicitly by the c0v launcher


def test_c0w_is_a_reference_anchored_contact_free_micro_lift_bridge():
    bridge = yaml.safe_load(BRIDGE_CONFIG.read_text())
    env = bridge["env"]
    contract = env["geometric_close_lift_bridge"]

    assert env["belt"] == {"speed": 0.06, "speed_noise": 0.25}
    assert env["target_lift"] == 0.005
    assert env["reward_set"] == "moving_cube_bridge_v1"
    assert contract == {
        "enabled": True,
        "close_steps": 5,
        "distance_threshold": 0.045,
        "min_close_fraction": 0.15,
        "max_close_fraction": 0.90,
        "prelift_max_height": 0.003,
        "target_lift": 0.005,
        "max_relative_drift": 0.008,
        "success_steps": 5,
    }
    assert bridge["init_checkpoint"] is None
    assert bridge["init_actor_scale"] is None
    assert bridge["loss"]["reference_checkpoint"] == "${init_checkpoint}"
    assert bridge["loss"]["reference_mean_kl_coeff"] > 0
    assert bridge["loss"]["freeze_actor_scale"] is True


def test_c0x_gates_progress_reward_and_relaxes_only_z_reference_kl():
    gated = yaml.safe_load(GATED_BRIDGE_CONFIG.read_text())
    contract = gated["env"]["geometric_close_lift_bridge"]

    assert gated["env"]["belt"] == {"speed": 0.06, "speed_noise": 0.25}
    assert contract["progress_reward"] is True
    assert contract["target_lift"] == 0.005
    assert contract["max_relative_drift"] == 0.008
    assert contract["tcp_progress_weight"] < contract["object_progress_weight"]
    assert gated["loss"]["reference_action_weights"] == [1.0, 1.0, 0.02, 1.0, 1.0, 1.0, 1.0]
    assert gated["loss"]["freeze_actor_scale"] is True
    assert gated["collector"]["total_frames"] == 6_000_000
