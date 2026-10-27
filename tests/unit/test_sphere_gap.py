import importlib.util
from pathlib import Path

import pytest
import torch

from pickplace.action_gates import GraspPoseCloseRetractOption
from pickplace.grasp_geometry import (
    deterministic_grasp_points,
    grasp_pose_deltas,
    nearest_grasp_delta,
    quat_xyzw_to_matrix,
    rotation_matrix_to_rotvec,
    sphere_grasp_frames,
)
from pickplace.envs.mdp.rewards import multiscale_pose_quality


def test_sphere_grasps_are_true_6d_and_have_requested_shape():
    centres = torch.tensor([[0.4, 0.3, 0.12], [0.5, 0.2, 0.12]])
    positions, rotations = sphere_grasp_frames(centres, num_candidates=40, tcp_height_offset=0.002)
    points = deterministic_grasp_points(positions, rotations, points_per_candidate=20, grasp_width=0.044)
    assert positions.shape == (2, 40, 3)
    assert rotations.shape == (2, 40, 3, 3)
    assert points.shape == (2, 40, 20, 3)
    assert torch.allclose(positions[..., 2], torch.full((2, 40), 0.122))
    assert torch.allclose(rotations.transpose(-1, -2) @ rotations, torch.eye(3).expand(2, 40, 3, 3), atol=1e-6)
    assert torch.all(torch.det(rotations) > 0.999)
    assert torch.allclose(rotations[..., 2], torch.tensor([0.0, 0.0, -1.0]).expand(2, 40, 3))


def test_canonical_sphere_grasps_repeat_one_physical_yaw_without_changing_shape():
    centres = torch.tensor([[0.4, 0.3, 0.12], [0.5, 0.2, 0.12]])
    positions, rotations = sphere_grasp_frames(
        centres, num_candidates=40, tcp_height_offset=0.002, canonical_yaw=0.0
    )
    points = deterministic_grasp_points(positions, rotations, points_per_candidate=20, grasp_width=0.044)
    assert positions.shape == (2, 40, 3)
    assert rotations.shape == (2, 40, 3, 3)
    assert points.shape == (2, 40, 20, 3)
    assert torch.allclose(rotations, rotations[:, :1].expand_as(rotations))
    assert torch.allclose(points, points[:, :1].expand_as(points))

    translation, rotation, nearest = nearest_grasp_delta(
        positions[:, 0],
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        positions,
        rotations,
    )
    assert torch.equal(nearest, torch.zeros(2, dtype=torch.long))
    assert translation.norm().item() < 1e-7
    assert rotation.norm().item() < 1e-5


def test_nearest_pose_delta_recognises_matching_topdown_tcp():
    centre = torch.tensor([[0.4, 0.3, 0.12]])
    positions, rotations = sphere_grasp_frames(centre, num_candidates=40, tcp_height_offset=0.002)
    # Candidate zero is a pi rotation around x: xyzw=(1,0,0,0).
    translation, rotation, nearest = nearest_grasp_delta(
        positions[:, 0], torch.tensor([[1.0, 0.0, 0.0, 0.0]]), positions, rotations
    )
    assert nearest.item() in (0, 20)  # parallel-jaw pi symmetry
    assert translation.norm().item() < 1e-7
    assert rotation.norm().item() < 1e-5


def test_all_pose_deltas_preserve_candidate_axis():
    centre = torch.tensor([[0.4, 0.3, 0.12]])
    positions, rotations = sphere_grasp_frames(centre, num_candidates=40, tcp_height_offset=0.002)
    translation, rotation = grasp_pose_deltas(
        positions[:, 0], torch.tensor([[1.0, 0.0, 0.0, 0.0]]), positions, rotations
    )
    assert translation.shape == (1, 40, 3)
    assert rotation.shape == (1, 40, 3)
    assert rotation[0, 0].norm().item() < 1e-5


def test_rotation_log_does_not_collapse_at_pi():
    matrix = torch.diag(torch.tensor([1.0, -1.0, -1.0])).unsqueeze(0)
    assert rotation_matrix_to_rotvec(matrix).norm(dim=-1).item() == pytest.approx(torch.pi, abs=1e-4)
    assert torch.allclose(quat_xyzw_to_matrix(torch.tensor([[1.0, 0.0, 0.0, 0.0]])), matrix)


def test_multiscale_pose_reward_has_far_field_gradient_and_near_precision():
    distances = torch.tensor([0.50, 0.40, 0.20, 0.10, 0.05, 0.02, 0.0], requires_grad=True)
    quality = multiscale_pose_quality(distances, torch.zeros_like(distances))
    quality.sum().backward()
    # Quality must improve monotonically all the way to the grasp, and the
    # former 40--50 cm dead zone must now have a meaningful negative slope.
    assert torch.all(quality[1:] > quality[:-1])
    assert distances.grad[0].item() < -0.1
    assert distances.grad[1].item() < -0.2
    assert quality[-1].item() == pytest.approx(1.0)


def test_multiscale_pose_reward_validates_scales_and_weight():
    value = torch.tensor([0.1])
    with pytest.raises(ValueError, match="scales"):
        multiscale_pose_quality(value, value, translation_far_scale=0.0)
    with pytest.raises(ValueError, match="near_weight"):
        multiscale_pose_quality(value, value, near_weight=1.1)


def test_pose_option_triggers_closes_then_retracts_without_speed_or_contact():
    option = GraspPoseCloseRetractOption(
        translation_threshold=0.012,
        rotation_threshold=0.15,
        approach_steps=1,
        close_steps=2,
        retract_steps=4,
        retract_height=0.04,
        action_scale=0.02,
    )
    action = torch.zeros(1, 7)
    pose_delta = torch.zeros(1, 6)
    ee_pos = torch.tensor([[0.4, 0.3, 0.12]])
    first, active = option(action, pose_delta, ee_pos)
    assert active.item() and first[0, -1].item() == 1.0
    option(action, pose_delta, ee_pos)
    third, active = option(action, pose_delta, ee_pos)
    assert active.item() and third[0, -1].item() == pytest.approx(-0.15)
    fourth, active = option(action, pose_delta, ee_pos)
    assert active.item() and fourth[0, 2].item() > 0.0


def test_replay_buffer_keeps_points_float16_but_samples_float32():
    path = Path(__file__).parents[2] / "pipeline" / "0_state_teacher" / "sac_utils.py"
    spec = importlib.util.spec_from_file_location("test_sac_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    replay = module.ReplayBuffer(16, [(3,), (40, 20, 3)], grasp_index=1)
    obs = [torch.randn(4, 3), torch.randn(4, 40, 20, 3)]
    replay.add(obs, torch.randn(4, 7), torch.randn(4, 1), obs, torch.zeros(4, 1))
    assert replay.obs[1].dtype == torch.float16
    sampled = replay.sample(3, "cpu")
    assert sampled[0][1].dtype == torch.float32
    assert sampled[1].shape == (3, 7)
