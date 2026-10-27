import importlib.util
from pathlib import Path

import pytest
import torch


UTILS = Path(__file__).parents[2] / "pipeline" / "0_state_teacher" / "utils.py"
SPEC = importlib.util.spec_from_file_location("teacher_reference_kl_test_utils", UTILS)
teacher_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(teacher_utils)


def test_gaussian_reference_kl_is_zero_for_identical_policy():
    loc = torch.tensor([[0.2, -0.1], [0.0, 0.4]])
    scale = torch.tensor([[0.5, 0.7], [0.5, 0.7]])

    full, mean = teacher_utils.gaussian_reference_kl(loc, scale, loc, scale)

    torch.testing.assert_close(full, torch.tensor(0.0))
    torch.testing.assert_close(mean, torch.tensor(0.0))


def test_gaussian_reference_kl_separates_mean_and_scale_drift():
    reference_loc = torch.zeros(1, 2)
    reference_scale = torch.ones(1, 2)
    loc = torch.tensor([[1.0, -2.0]], requires_grad=True)
    scale = torch.tensor([[2.0, 0.5]], requires_grad=True)

    full, mean = teacher_utils.gaussian_reference_kl(loc, scale, reference_loc, reference_scale)

    expected_mean = torch.tensor(0.5 * (1.0**2 + 2.0**2))
    expected_full = (
        torch.log(reference_scale / scale.detach())
        + 0.5 * ((scale.detach() / reference_scale).square() + loc.detach().square() - 1.0)
    ).sum()
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(full, expected_full)
    full.backward()
    assert loc.grad is not None and torch.isfinite(loc.grad).all()
    assert scale.grad is not None and torch.isfinite(scale.grad).all()


def test_gaussian_reference_kl_can_relax_only_one_action_dimension():
    reference_loc = torch.zeros(1, 3)
    reference_scale = torch.ones(1, 3)
    loc = torch.ones(1, 3)
    scale = torch.ones(1, 3)

    full, mean = teacher_utils.gaussian_reference_kl(
        loc,
        scale,
        reference_loc,
        reference_scale,
        action_weights=torch.tensor([1.0, 1.0, 0.02]),
    )

    expected = torch.tensor(0.5 * 2.02)
    torch.testing.assert_close(full, expected)
    torch.testing.assert_close(mean, expected)


def test_gaussian_reference_kl_rejects_bad_action_weights():
    values = torch.zeros(1, 3)
    scale = torch.ones(1, 3)
    with pytest.raises(ValueError, match="expected"):
        teacher_utils.gaussian_reference_kl(values, scale, values, scale, action_weights=[1.0, 1.0])
