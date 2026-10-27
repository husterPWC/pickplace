import torch

from pickplace.envs.mdp.observations import (
    gripper_blocked_between_limits,
    gripper_close_fraction,
)


def test_close_fraction_preserves_franka_direction():
    q = torch.tensor([[0.04], [0.02], [0.0]])
    assert torch.allclose(gripper_close_fraction(q, 0.04, 0.0), torch.tensor([0.0, 0.5, 1.0]))


def test_close_fraction_supports_ag95_reversed_direction():
    q = torch.tensor([[0.0], [0.4715], [0.943]])
    assert torch.allclose(gripper_close_fraction(q, 0.0, 0.943), torch.tensor([0.0, 0.5, 1.0]))


def test_blocked_between_limits_is_direction_independent():
    q = torch.tensor([[0.0], [0.30], [0.943]])
    expected = torch.tensor([False, True, False])
    assert torch.equal(gripper_blocked_between_limits(q, 0.0, 0.943, 0.004), expected)
    assert torch.equal(gripper_blocked_between_limits(q, 0.943, 0.0, 0.004), expected)
