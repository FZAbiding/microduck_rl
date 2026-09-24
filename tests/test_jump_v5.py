"""Jump-V5 exact-height and curriculum regression tests (CPU)."""
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.jump_control import JumpController
from mjlab_microduck.jump_curriculum import (
    V5_HEIGHTS,
    configure_v5,
    v5_metrics_pass,
    v5_reward_regressed,
)
from mjlab_microduck.jump_runner import restore_jump_contract
from mjlab_microduck.jump_v5 import (
    add_baseline,
    finish_block,
    initial_state,
    plan_block,
)


def metrics(**overrides):
    result = {
        "success_rate": .96,
        "height_rate": .96,
        "takeoff_rate": .96,
        "peak_delta_p10": .031,
        "heading_change_p90_deg": 9.,
        "heading_change_max_deg": 19.,
        "max_heading_error_p90_deg": 9.,
        "max_heading_error_max_deg": 19.,
        "drift_p90_m": .039,
    }
    result.update(overrides)
    return result


def standing(rate=.96):
    return {"standing_rate": rate}


def test_v5_required_delta_is_exact_and_legacy_discount_is_preserved():
    exact = JumpController(1, target_delta=.04, required_delta=.04)
    assert exact.acceptance_delta == pytest.approx(.04)

    legacy = JumpController(1, target_delta=.04)
    assert legacy.acceptance_delta == pytest.approx(.035)
    legacy.target_delta = .03
    assert legacy.acceptance_delta == pytest.approx(.025)


def test_restore_missing_required_delta_keeps_old_checkpoint_contract():
    controller = JumpController(1, target_delta=.01, required_delta=.01)
    term = SimpleNamespace(protocol=2)
    restore_jump_contract(
        controller,
        {"stand_z": .115, "target_delta": .04, "command_protocol": 2},
        term,
    )
    assert controller.required_delta is None
    assert controller.acceptance_delta == pytest.approx(.035)
    assert term.protocol == 2

    restore_jump_contract(
        controller,
        {
            "stand_z": .116,
            "target_delta": .05,
            "required_delta": .05,
            "command_protocol": 2,
        },
        term,
    )
    assert controller.stand_z == pytest.approx(.116)
    assert controller.acceptance_delta == pytest.approx(.05)


def test_v5_cfg_is_nominal_protocol_two_with_fixed_reward_signs():
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg

    base = make_microduck_jump_env_cfg(
        nominal_bootstrap=False, command_protocol=2
    )
    cfg = configure_v5(
        base, target_delta=.05, height_weight=1.5,
        heading_weight=-.20, yaw_rate_weight=-.02, head_bias_weight=.5,
    )
    assert cfg.commands["twist"].protocol == 2
    assert cfg.rewards["jump_height_progress"].weight == pytest.approx(1.5)
    assert cfg.rewards["jump_heading_error"].weight == pytest.approx(-.20)
    assert cfg.rewards["jump_yaw_rate"].weight == pytest.approx(-.02)
    assert cfg.rewards["jump_head_bias"].weight == pytest.approx(.5)
    assert cfg.rewards["action_rate_l2"].weight == pytest.approx(-.02)
    assert cfg.rewards["body_ang_vel"].weight == base.rewards["body_ang_vel"].weight
    assert cfg.rewards["angular_momentum"].weight == base.rewards["angular_momentum"].weight
    assert cfg.rewards["jump_landing_impact"].weight == base.rewards["jump_landing_impact"].weight
    assert set(cfg.events) == {
        "reset_base", "reset_robot_joints", "expand_bam_friction_fields",
        "reset_action_history", "jump_reset_state",
    }
    assert all(not group.enable_corruption for group in cfg.observations.values())
    assert not cfg.curriculum
    assert base.rewards["jump_height_progress"].weight == 1.0


def test_v5_cfg_rejects_non_contract_reward_or_height_values():
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg

    base = make_microduck_jump_env_cfg(
        nominal_bootstrap=False, command_protocol=2
    )
    with pytest.raises(ValueError):
        configure_v5(base, target_delta=.055)
    with pytest.raises(ValueError):
        configure_v5(base, height_weight=1.25)
    with pytest.raises(ValueError):
        configure_v5(base, heading_weight=-.3, yaw_rate_weight=-.02)
    with pytest.raises(ValueError):
        configure_v5(base, head_bias_weight=1.0)


def test_v5_strict_gate_uses_p10_but_head_metrics_are_record_only():
    jump = metrics(head_dynamic_p90_deg=180., head_complete_p90_deg=180.)
    assert v5_metrics_pass(jump, standing(), .03)
    assert not v5_metrics_pass({**jump, "peak_delta_p10": .0299}, standing(), .03)
    assert not v5_metrics_pass({**jump, "max_heading_error_max_deg": 20.1}, standing(), .03)
    assert not v5_metrics_pass({**jump, "drift_p90_m": .0401}, standing(), .03)


@pytest.mark.parametrize(
    ("jump", "stand", "expected"),
    [
        (metrics(takeoff_rate=.90), standing(), True),
        (metrics(), standing(.90), True),
        (metrics(max_heading_error_p90_deg=14.1), standing(), True),
        (metrics(heading_change_p90_deg=14.1), standing(), True),
        (metrics(takeoff_rate=.91), standing(.91), False),
    ],
)
def test_v5_reward_change_rollback_thresholds(jump, stand, expected):
    assert v5_reward_regressed(jump, stand, metrics(), standing()) is expected


def test_v5_height_reward_regression_restores_checkpoint_and_weight():
    state = initial_state("initial.pt")
    add_baseline(state, metrics(), standing())
    assert plan_block(state) == 125
    finish_block(state, "head.pt", 125, metrics(), standing())
    assert state["bootstrap_phase"] == 1

    assert plan_block(state) == 125
    assert state["height_weight"] == 1.5
    assert state["pending_reward_change"]["kind"] == "height_1p5"
    status = finish_block(
        state, "regressed.pt", 125,
        metrics(takeoff_rate=.90), standing(),
    )
    assert status == "rollback_height_1p5"
    assert state["checkpoint"] == "head.pt"
    assert state["height_weight"] == 1.0
    assert state["new_updates"] == 250
    assert state["streak"] == 0


def test_v5_heading_tier_is_only_added_when_strict_heading_still_fails():
    bad_heading = metrics(max_heading_error_p90_deg=12.)
    state = initial_state("initial.pt")
    add_baseline(state, bad_heading, standing())
    assert plan_block(state) == 125
    finish_block(state, "head.pt", 125, bad_heading, standing())
    assert plan_block(state) == 125
    finish_block(state, "height.pt", 125, bad_heading, standing())
    assert plan_block(state) == 125
    assert state["heading_weight"] == pytest.approx(-.30)
    assert state["yaw_rate_weight"] == pytest.approx(-.03)
    assert state["pending_reward_change"]["kind"] == "strong_heading"


def test_v5_budget_and_slice_caps_are_hard_and_blocks_never_exceed_250():
    state = initial_state("checkpoint.pt")
    state.update(
        new_updates=4990, height_index=len(V5_HEIGHTS)-1,
        slice_updates=1490, bootstrap_phase=3,
    )
    state["last_selected_evaluation"] = {
        "jump": metrics(), "standing": standing(), "checkpoint": "checkpoint.pt"
    }
    assert plan_block(state) == 10
    finish_block(state, "last.pt", 10, metrics(peak_delta_p10=.051), standing())
    assert state["new_updates"] == 5000
    assert plan_block(state) == 0

    capped = initial_state("checkpoint.pt")
    capped.update(new_updates=100, slice_updates=748, bootstrap_phase=3)
    capped["last_selected_evaluation"] = state["last_selected_evaluation"]
    assert plan_block(capped) == 2


def test_protocol_two_command_is_device_consistent():
    cpu = JumpController(2)
    cpu.heading_error[:] = torch.tensor([-.5, .5])
    expected = cpu.command(2)
    assert expected[:, 0].eq(0).all()
    assert expected[:, 1].eq(0).all()
    assert expected[:, 2].tolist() == pytest.approx([-.1, .1])
    if torch.cuda.is_available():
        gpu = JumpController(2, device="cuda:0")
        gpu.heading_error[:] = torch.tensor([-.5, .5], device="cuda:0")
        assert torch.allclose(expected, gpu.command(2).cpu())
