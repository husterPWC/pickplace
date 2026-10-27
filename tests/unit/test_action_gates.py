import pytest
import torch

from tensordict import TensorDict

from pickplace.action_gates import (
    CloseRetractOption,
    GeometricMicroLiftOption,
    GraspHoldGate,
    make_stable_sphere_pick_place_skill,
    make_stable_sphere_residual_skill,
)


def test_gate_suppresses_only_upward_motion_and_forces_close_for_exact_window():
    gate = GraspHoldGate(hold_steps=3)
    action = torch.tensor([[0.2, -0.3, 0.8, 0.4, 0.5, 0.6, 1.0]])
    held = torch.tensor([[True]])

    for _ in range(3):
        out, active = gate(action, held)
        assert active.tolist() == [True]
        assert out[0, 2].item() == 0.0
        assert out[0, -1].item() == -1.0
        assert torch.equal(out[0, :2], action[0, :2])
        assert torch.equal(out[0, 3:6], action[0, 3:6])

    out, active = gate(action, held)
    assert active.tolist() == [False]
    assert torch.equal(out, action)


def test_gate_allows_downward_correction_and_restarts_after_grasp_loss():
    gate = GraspHoldGate(hold_steps=2, max_upward_dz=0.1)
    down = torch.tensor([[0.0, 0.0, -0.4, 0.0, 0.0, 0.0, 0.5]])
    up = down.clone()
    up[:, 2] = 0.7

    out, _ = gate(down, torch.tensor([True]))
    assert out[0, 2].item() == pytest.approx(-0.4)
    gate(up, torch.tensor([True]))
    gate(up, torch.tensor([False]))
    out, active = gate(up, torch.tensor([True]))
    assert active.tolist() == [True]
    assert out[0, 2].item() == pytest.approx(0.1)


def test_gate_rejects_non_ee_action_shape():
    gate = GraspHoldGate(hold_steps=1)
    with pytest.raises(ValueError, match="7-D"):
        gate(torch.zeros(2, 8), torch.ones(2, 1, dtype=torch.bool))


def test_gate_follows_object_xy_and_freezes_rotation_during_hold():
    gate = GraspHoldGate(
        hold_steps=2, follow_object_xy=True, control_dt=0.02, action_scale=0.5,
        follow_gain=0.5, max_follow_speed=0.15, freeze_rotation=True,
    )
    action = torch.tensor([[0.8, -0.7, 0.9, 0.4, -0.5, 0.6, 1.0]])
    food = torch.tensor([[0.40, 0.30, 0.10]])
    ee = torch.tensor([[0.39, 0.31, 0.10]])
    velocity = torch.tensor([[0.04, 0.0, 0.0]])

    out, active = gate(action, torch.tensor([True]), food, ee, velocity)
    assert active.tolist() == [True]
    assert out[0, 0].item() == pytest.approx(0.04 * 0.02 / 0.5)
    assert out[0, 1].item() == pytest.approx(0.0)
    assert torch.equal(out[0, 3:6], torch.zeros(3))

    # Object moved 2 mm farther ahead than the anchored relative pose: add a
    # 1 mm proportional correction plus the 0.8 mm velocity feed-forward.
    moved = food.clone()
    moved[:, 0] += 0.002
    out, _ = gate(action, torch.tensor([True]), moved, ee, velocity)
    assert out[0, 0].item() == pytest.approx((0.0008 + 0.001) / 0.5, abs=1e-7)


def test_follow_gate_requires_object_state():
    gate = GraspHoldGate(hold_steps=1, follow_object_xy=True)
    with pytest.raises(ValueError, match="requires object_pos"):
        gate(torch.zeros(1, 7), torch.tensor([True]))


def test_enabled_gate_requires_native_ee_actions_and_grasp_observation():
    from pickplace.config import build_cell_env_cfg

    with pytest.raises(ValueError, match="privileged_information=true"):
        build_cell_env_cfg({"grasp_hold_gate": {"hold_steps": 5}})
    with pytest.raises(ValueError, match="native ee_delta_pose"):
        build_cell_env_cfg({
            "action_mode": "joint_pos",
            "privileged_information": True,
            "grasp_hold_gate": {"hold_steps": 5},
        })


def test_close_retract_option_uses_geometry_and_smoothly_accumulates_height():
    option = CloseRetractOption(
        steps=4, retract_height=0.04, trigger_distance=0.05, trigger_gripper=-0.5,
        control_dt=0.02, action_scale=0.5, follow_gain=0.5,
        max_follow_speed=0.15, xy_blend=1.0,
    )
    action = torch.tensor([[0.8, -0.8, -0.5, 0.1, 0.2, 0.3, -1.0]])
    food = torch.tensor([[0.40, 0.20, 0.10]])
    ee = torch.tensor([[0.39, 0.20, 0.10]])
    velocity = torch.tensor([[0.04, 0.0, 0.0]])
    dz = []
    for _ in range(4):
        out, active, _ = option(action, food, ee, velocity)
        assert active.tolist() == [True]
        assert out[0, -1].item() == -1.0
        assert out[0, 0].item() == pytest.approx(0.04 * 0.02 / 0.5)
        dz.append(out[0, 2].item() * 0.5)
    assert sum(dz) == pytest.approx(0.04, abs=1e-7)
    out, active, _ = option(action, food, ee, velocity)
    assert active.tolist() == [False]
    assert torch.equal(out, action)


def test_close_retract_option_does_not_retrigger_until_open_or_miss():
    option = CloseRetractOption(
        steps=1, retract_height=0.01, trigger_distance=0.05, trigger_gripper=-0.5,
        control_dt=0.02, action_scale=0.5, follow_gain=0.0,
        max_follow_speed=0.15, xy_blend=0.0,
    )
    close = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
    food = torch.tensor([[0.40, 0.20, 0.10]])
    ee = torch.tensor([[0.39, 0.20, 0.10]])
    velocity = torch.zeros(1, 3)
    _, active, _ = option(close, food, ee, velocity)
    assert active.tolist() == [True]
    _, active, _ = option(close, food, ee, velocity)
    assert active.tolist() == [False]
    opened = close.clone()
    opened[:, -1] = 1.0
    option(opened, food, ee, velocity)
    _, active, _ = option(close, food, ee, velocity)
    assert active.tolist() == [True]


def test_geometric_micro_lift_waits_for_stable_close_then_tracks_height_without_velocity():
    option = GeometricMicroLiftOption(
        close_steps=3,
        lift_steps=4,
        hold_steps=2,
        lift_height=0.008,
        distance_threshold=0.045,
        min_close_fraction=0.15,
        max_close_fraction=0.90,
        open_pos=0.04,
        closed_pos=0.0,
        action_scale=0.5,
        max_z_step=0.01,
    )
    policy = torch.tensor([[0.3, -0.2, -0.6, 0.1, 0.2, 0.3, 1.0]])
    offset = torch.tensor([[0.01, 0.0, 0.0]])
    ee = torch.tensor([[0.4, 0.2, 0.1]])
    fingers = torch.tensor([[0.02, 0.02]])  # 50% closed: cube obstructs the fingers

    # First sample initializes the encoder history; the next two establish
    # that the partial aperture has actually stopped moving.
    for _ in range(3):
        out, active, _ = option(policy, offset, ee, fingers)
        assert active.tolist() == [False]
        assert torch.equal(out, policy)

    commanded_dz = []
    for _ in range(6):
        out, active, _ = option(policy, offset, ee, fingers)
        assert active.tolist() == [True]
        assert out[0, -1].item() == -1.0
        assert torch.equal(out[0, 3:6], torch.zeros(3))
        dz = out[0, 2].item() * 0.5
        commanded_dz.append(dz)
        ee[:, 2] += dz
    assert sum(commanded_dz) == pytest.approx(0.008, abs=1e-7)

    out, active, _ = option(policy, offset, ee, fingers)
    assert active.tolist() == [False]
    assert torch.equal(out, policy)


def test_geometric_micro_lift_position_only_xy_servo_preserves_latched_offset():
    option = GeometricMicroLiftOption(
        close_steps=1,
        lift_steps=2,
        hold_steps=0,
        lift_height=0.002,
        distance_threshold=0.045,
        min_close_fraction=0.15,
        max_close_fraction=0.90,
        open_pos=0.04,
        closed_pos=0.0,
        action_scale=0.5,
        xy_gain=0.5,
        max_xy_step=0.002,
        max_z_step=0.001,
    )
    action = torch.zeros(1, 7)
    offset = torch.tensor([[0.01, -0.01, 0.0]])
    ee = torch.tensor([[0.4, 0.2, 0.1]])
    fingers = torch.tensor([[0.02, 0.02]])
    option(action, offset, ee, fingers)  # initialize encoder history
    option(action, offset, ee, fingers)  # stable partial aperture triggers

    moved = offset + torch.tensor([[0.002, -0.004, 0.0]])
    out, active, _ = option(action, moved, ee, fingers)
    assert active.tolist() == [True]
    # 0.5 gain gives [+1, -2] mm, divided by the 0.5 m native action scale.
    assert out[0, 0].item() == pytest.approx(0.001 / 0.5)
    assert out[0, 1].item() == pytest.approx(-0.002 / 0.5)


def test_enabled_geometric_micro_lift_requires_position_history():
    from pickplace.config import build_cell_env_cfg

    with pytest.raises(ValueError, match="position_history_steps"):
        build_cell_env_cfg({"geometric_micro_lift_option": {"enabled": True}})


def test_sphere_residual_skill_resets_finished_rows_on_native_autoreset():
    transform = make_stable_sphere_residual_skill(
        translation_residual_limit=0.1,
        rotation_residual_limit=0.05,
    )
    baseline = transform.controller.baseline
    baseline.phase = torch.tensor([baseline.DONE, baseline.LIFT])
    baseline.phase_step = torch.tensor([99, 12])
    baseline.alignment_count = torch.tensor([7, 3])
    baseline.lift_anchor_z = torch.tensor([0.42, 0.31])
    reset_mask = torch.tensor([[True], [False]])
    current = TensorDict({"_reset": reset_mask}, batch_size=[2])
    reset = TensorDict({}, batch_size=[2])

    assert transform._reset_on_native_autoreset(current, reset) is reset
    assert baseline.phase.tolist() == [baseline.HOVER, baseline.LIFT]
    assert baseline.phase_step.tolist() == [0, 12]
    assert baseline.alignment_count.tolist() == [0, 3]
    assert baseline.lift_anchor_z.tolist() == pytest.approx([0.0, 0.31])


def test_sphere_pick_place_transform_hands_done_grasp_to_vertical_clear():
    transform = make_stable_sphere_pick_place_skill(
        transport_config={"stable_steps": 1},
        translation_residual_limit=0.0,
        rotation_residual_limit=0.0,
        action_scale=0.02,
    )
    controller = transform.controller
    action = torch.zeros(1, 7)
    controller.grasp.baseline._ensure(action)
    controller.grasp.baseline.phase.fill_(controller.grasp.baseline.DONE)
    td = TensorDict(
        {
            "action": action,
            "grasp_guidance": {"candidate_pose_deltas": torch.zeros(1, 1, 6)},
            "proprio": {
                "ee_pos": torch.tensor([[0.45, 0.30, 0.10]]),
                "gripper_pos": torch.full((1, 2), 0.02),
            },
            "belt": {"bowl_pos": torch.tensor([[0.45, 0.0, 0.0]])},
        },
        batch_size=[1],
    )

    assert transform._inv_call(td) is td
    assert controller.phase.item() == controller.CLEAR
    assert td["action"][0, 2].item() > 0.0
    assert td["action"][0, 1].item() == pytest.approx(0.0)
    assert td["action"][0, -1].item() == pytest.approx(-0.15)
