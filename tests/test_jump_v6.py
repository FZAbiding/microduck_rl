"""Jump-V6 precision-vertical contract regression tests."""
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.jump_control import JumpController
from mjlab_microduck.jump_curriculum import (
    V6_HEIGHTS,
    V6_MAX_UPDATES,
    V6_REWARD_WEIGHTS,
    configure_v6,
    v6_metrics_pass,
)
from mjlab_microduck.tasks.symmetry import microduck_mirror_actions


def test_v6_reaches_ten_cm_and_uses_exact_height_contract():
    assert V6_HEIGHTS[-1] == pytest.approx(.10)
    assert V6_MAX_UPDATES == 12000
    controller = JumpController(1, target_delta=.10, required_delta=.10)
    assert controller.acceptance_delta == pytest.approx(.10)


def test_request_captures_and_reset_clears_planar_target():
    controller = JumpController(2)
    controller.ready_time[:] = 1.0
    xy = torch.tensor([[.12, -.03], [.04, .08]])
    accepted = controller.press(torch.tensor([True, False]), current_xy=xy)
    assert accepted.tolist() == [True, False]
    assert torch.allclose(controller.target_xy, torch.tensor([[.12, -.03], [0., 0.]]))
    controller.reset(torch.tensor([0]))
    assert controller.target_xy.eq(0).all()


def test_action_mirror_is_an_involution_and_rejects_wrong_shape():
    actions = torch.randn(8, 14)
    assert torch.allclose(microduck_mirror_actions(microduck_mirror_actions(actions)), actions)
    with pytest.raises(ValueError):
        microduck_mirror_actions(torch.zeros(2, 13))
    from mjlab_microduck.tasks.symmetry import _cache
    _cache.clear()
    with torch.inference_mode():
        microduck_mirror_actions(torch.zeros(2, 14))
    differentiable = torch.randn(2, 14, requires_grad=True)
    microduck_mirror_actions(differentiable).sum().backward()
    assert differentiable.grad is not None


def test_v6_cfg_adds_dense_precision_costs_without_mutating_v5_base():
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg

    base = make_microduck_jump_env_cfg(nominal_bootstrap=False, command_protocol=2)
    original_names = set(base.rewards)
    cfg = configure_v6(base, target_delta=.10)
    assert set(base.rewards) == original_names
    assert cfg.commands["twist"].protocol == 2
    assert not cfg.curriculum
    assert all(not group.enable_corruption for group in cfg.observations.values())
    for name in (
        "jump_planar_displacement",
        "jump_planar_velocity",
        "jump_upright",
        "jump_post_landing_pose",
        "jump_launch_action_symmetry",
    ):
        assert cfg.rewards[name].weight == pytest.approx(V6_REWARD_WEIGHTS[name])
        assert cfg.rewards[name].weight < 0
    assert cfg.rewards["jump_height_progress"].weight > 0
    assert cfg.rewards["jump_head_bias"].weight > 0


def test_v6_gate_requires_height_and_precision_state_restore():
    jump = {
        "success_rate": .96,
        "height_rate": .96,
        "takeoff_rate": .96,
        "peak_delta_p10": .101,
        "heading_change_p90_deg": 4.9,
        "heading_change_max_deg": 9.9,
        "max_heading_error_p90_deg": 4.9,
        "max_heading_error_max_deg": 9.9,
        "drift_p90_m": .019,
        "final_drift_p90_m": .009,
        "final_tilt_p90_deg": 2.9,
        "final_pose_l1_p90_rad": .099,
        "final_horizontal_speed_p90_m_s": .029,
    }
    standing = {"standing_rate": .96}
    assert v6_metrics_pass(jump, standing, .10)
    for key, bad in (
        ("peak_delta_p10", .0999),
        ("drift_p90_m", .0201),
        ("final_drift_p90_m", .0101),
        ("heading_change_p90_deg", 5.1),
        ("final_pose_l1_p90_rad", .101),
    ):
        assert not v6_metrics_pass({**jump, key: bad}, standing, .10)


def test_v6_rejects_unregistered_height():
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg

    base = make_microduck_jump_env_cfg(nominal_bootstrap=False, command_protocol=2)
    with pytest.raises(ValueError):
        configure_v6(base, target_delta=.055)


def test_v6_state_machine_advances_only_after_two_passes_and_hard_caps():
    from mjlab_microduck.jump_v6 import (
        add_baseline, finish_block, initial_state, plan_block, target_delta
    )

    good = {
        "success_rate": 1., "height_rate": 1., "takeoff_rate": 1.,
        "peak_delta_p10": .031, "heading_change_p90_deg": 1.,
        "heading_change_max_deg": 2., "max_heading_error_p90_deg": 1.,
        "max_heading_error_max_deg": 2., "drift_p90_m": .01,
        "final_drift_p90_m": .005, "final_tilt_p90_deg": 1.,
        "final_pose_l1_p90_rad": .05,
        "final_horizontal_speed_p90_m_s": .01,
    }
    stand = {"standing_rate": 1.}
    state = initial_state("start.pt")
    add_baseline(state, good, stand)
    assert plan_block(state) == 125
    assert finish_block(state, "a.pt", 125, good, stand) == "continue"
    assert target_delta(state) == pytest.approx(.03)
    assert plan_block(state) == 125
    assert finish_block(state, "b.pt", 125, good, stand) == "advanced"
    assert target_delta(state) == pytest.approx(.04)

    state.update(new_updates=11990, height_index=len(V6_HEIGHTS)-1,
                 slice_updates=990, streak=0, status="continue")
    assert plan_block(state) == 10
    with pytest.raises(ValueError):
        finish_block(state, "overflow.pt", 11, good, stand)
