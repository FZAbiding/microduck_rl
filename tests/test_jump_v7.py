"""Jump-V7 recovery, sequencing, and budget contract tests."""
from __future__ import annotations

import math

import pytest
import torch

from mjlab_microduck.jump_control import JumpController
from mjlab_microduck.jump_curriculum import (
    V7_MAX_UPDATES,
    V7_REWARD_WEIGHTS,
    configure_v7,
    v7_branch_key,
    v7_metrics_pass,
)
from mjlab_microduck.jump_v7 import (
    finish_block,
    initial_state,
    plan_block,
    record_smoke,
)


def sample(c, step, *, feet=(True, True), z=.115, vz=0., body=False,
           tilt_deg=0., joint_l1=0., horizontal_speed=0., yaw_rate=0.):
    tilt = math.radians(tilt_deg)
    quat = torch.tensor([[math.cos(tilt / 2), math.sin(tilt / 2), 0., 0.]])
    omega = torch.tensor([[0., 0., yaw_rate]])
    c.update(
        step,
        torch.tensor([z]),
        quat,
        torch.tensor([vz]),
        omega,
        torch.tensor([feet]),
        torch.tensor([body]),
        torch.tensor([10.]),
        torch.tensor([joint_l1]),
        torch.tensor([horizontal_speed]),
    )


def arm_v7():
    c = JumpController(
        1, dt=.02, stand_z=.115, target_delta=.03,
        required_delta=.03, policy_version=7,
    )
    for step in range(100):
        sample(c, step)
    assert c.ready_time.item() == pytest.approx(2.0)
    assert c.press(torch.tensor([True])).item()
    return c


def takeoff_and_land(c, start):
    sample(c, start, feet=(False, False), z=.145, vz=.2)
    sample(c, start + 1, feet=(False, False), z=.145, vz=-.2)
    sample(c, start + 2, feet=(True, False), z=.12)
    sample(c, start + 3, feet=(True, True))
    assert c.landed.item()
    return start + 4


def recover(c, start, seconds=5.0):
    steps = round(seconds / c.dt)
    payments = 0
    early_success = 0
    for step in range(start, start + steps):
        sample(c, step)
        payments += int(c.complete_event.item())
        early_success += int(c.success_event.item())
    return start + steps, payments, early_success


def good_metrics(sequence='single'):
    common = {
        'failure_rate': 0.,
        'body_contact_rate': 0.,
        'invalid_rate': 0.,
        'peak_delta_p10': .03,
        'final_drift_p90_m': .01,
        'final_drift_max_m': .02,
        'heading_change_p90_deg': 3.,
        'heading_change_max_deg': 5.,
        'recovery_tilt_p90_deg': 3.,
        'recovery_tilt_max_deg': 5.,
        'recovery_pose_l1_p90_rad': .08,
        'final_horizontal_speed_max_m_s': .03,
    }
    if sequence == 'single':
        return {**common, 'success_rate': .95}
    return {**common, 'sequence_success_rate': .95, 'jump_success_rate': .98}


def test_v7_repeat_ready_gate_requires_every_condition_for_two_seconds():
    variants = (
        {'feet': (True, False)},
        {'tilt_deg': 3.01},
        {'joint_l1': .081},
        {'horizontal_speed': .031},
        {'yaw_rate': .101},
    )
    for bad in variants:
        c = JumpController(1, dt=.02, policy_version=7)
        # The first request uses the source-policy bootstrap stance. Once any
        # request has been accepted, every subsequent request uses V7 strictness.
        c.accepted[:] = True
        for step in range(99):
            sample(c, step, z=.115)
        assert not c.press(torch.tensor([True])).item()
        sample(c, 99, z=.115, **bad)
        assert c.ready_time.item() == 0.
        for step in range(100, 200):
            sample(c, step, z=.115)
        assert c.press(torch.tensor([True])).item()

    body_contact = JumpController(1, dt=.02, policy_version=7)
    sample(body_contact, 0, z=.115, body=True)
    assert body_contact.invalid.item()
    for step in range(1, 101):
        sample(body_contact, step, z=.115)
    assert not body_contact.press(torch.tensor([True])).item()


def test_v7_has_no_early_success_and_completes_once_after_five_seconds():
    c = arm_v7()
    step = takeoff_and_land(c, 100)
    step, payments, early_success = recover(c, step, 4.96)
    assert not c.complete.item()
    assert payments == 0
    assert early_success == 0
    sample(c, step)
    assert c.complete.item()
    assert c.complete_event.item() == 1.
    assert c.success_event.item() == 0.
    sample(c, step + 1)
    assert c.complete_event.item() == 0.


def test_v7_completion_does_not_hide_later_fall():
    c = arm_v7()
    step = takeoff_and_land(c, 100)
    step, _, _ = recover(c, step, 5.0)
    assert c.complete.item()
    sample(c, step, body=True)
    assert c.invalid.item() and c.fallen.item()
    assert c.failure_event.item() == 1.


def test_v7_rejects_next_request_before_recovery_and_catches_third_jump_fall():
    c = arm_v7()
    step = takeoff_and_land(c, 100)
    for _ in range(2):
        sample(c, step)
        step += 1
    assert not c.press(torch.tensor([True])).item()
    step, payments, _ = recover(c, step, 4.96)
    assert payments == 1
    assert c.press(torch.tensor([True])).item()
    step = takeoff_and_land(c, step)
    step, payments, _ = recover(c, step, 5.0)
    assert payments == 1
    assert c.press(torch.tensor([True])).item()
    sample(c, step, feet=(False, False), z=.145)
    sample(c, step + 1, feet=(False, False), z=.145)
    sample(c, step + 2, tilt_deg=45.01)
    assert c.invalid.item() and c.fallen.item()


def test_v7_failure_and_timeout_are_distinct_one_shot_events():
    c = arm_v7()
    for step in range(100, 174):
        sample(c, step)
    assert not c.timeout.item()
    sample(c, 174)
    assert c.timeout.item() and c.timeout_event.item() == 1.
    assert c.failure_event.item() == 0.
    sample(c, 175)
    assert c.timeout_event.item() == 0.

    c = arm_v7()
    sample(c, 100, body=True)
    assert c.failure_event.item() == 1.
    assert c.timeout_event.item() == 0.
    sample(c, 101, body=True)
    assert c.failure_event.item() == 0.


def test_v7_phase_masks_cover_launch_flight_touchdown_recovery_and_idle():
    c = arm_v7()
    assert c.phase_mask('launch').item()
    sample(c, 100, feet=(False, False), z=.145)
    sample(c, 101, feet=(False, False), z=.145)
    assert c.phase_mask('flight').item()
    sample(c, 102, feet=(True, False), z=.12)
    assert c.phase_mask('touchdown').item()
    sample(c, 103)
    assert c.phase_mask('recovery').item()
    assert c.phase_mask('post_contact').item()
    step, _, _ = recover(c, 104, 5.0)
    assert c.phase_mask('idle').item()
    assert not c.phase_mask('post_contact').item()


def test_v7_cfg_has_fixed_rewards_phase_weights_and_episode_mix():
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg

    base = make_microduck_jump_env_cfg(
        nominal_bootstrap=False, command_protocol=2
    )
    cfg = configure_v7(base)
    assert base.episode_length_s == 6.
    assert cfg.episode_length_s == 24.
    command = cfg.commands['twist']
    assert command.protocol == 2 and command.policy_version == 7
    assert command.standing_probability == pytest.approx(.20)
    assert command.triple_probability == pytest.approx(.30)
    assert command.request_times == (7., 14., 21.)
    assert command.request_jitter_s == pytest.approx(.5)
    assert not cfg.curriculum
    assert all(not group.enable_corruption for group in cfg.observations.values())
    assert 'jump_success_score' not in cfg.rewards
    assert 'jump_launch_velocity' not in cfg.rewards
    for name, weight in V7_REWARD_WEIGHTS.items():
        assert cfg.rewards[name].weight == pytest.approx(weight)
    assert cfg.rewards['jump_planar_displacement_launch'].params['phase'] == 'launch_flight'
    assert cfg.rewards['jump_planar_displacement_recovery'].params['phase'] == 'post_contact'


def test_v7_acceptance_gate_and_lexicographic_branch_order():
    single = good_metrics('single')
    triple = good_metrics('triple')
    assert v7_metrics_pass(single, triple)
    assert not v7_metrics_pass({**single, 'failure_rate': .01}, triple)
    assert not v7_metrics_pass(single, {**triple, 'jump_success_rate': .979})
    unsafe_but_accurate = {
        **triple, 'failure_rate': .01, 'sequence_success_rate': 1.,
        'final_drift_p90_m': 0., 'heading_change_p90_deg': 0.,
    }
    safe_lower_success = {**triple, 'sequence_success_rate': .90}
    assert v7_branch_key(single, safe_lower_success) > v7_branch_key(
        single, unsafe_but_accurate
    )


def test_v7_two_branch_budget_selection_and_two_boundary_streak():
    state = initial_state('v5.pt', 'v6.pt')
    record_smoke(state)
    assert state['new_updates'] == 5
    for name in ('v5', 'v5', 'v6', 'v6'):
        branch, count = plan_block(state)
        assert (branch, count) == (name, 250)
        single = good_metrics('single')
        triple = good_metrics('triple')
        if name == 'v5':
            triple = {**triple, 'sequence_success_rate': .90}
        finish_block(
            state, name, f'{name}-{state["branches"][name]["updates"] + 250}.pt',
            count, single, triple,
        )
    assert state['selected_branch'] == 'v6'
    assert state['new_updates'] == 1005
    for expected_status in ('continue', 'candidate_passed'):
        name, count = plan_block(state)
        assert name == 'v6' and count == 250
        status = finish_block(
            state, name, f'v6-{state["new_updates"] + count}.pt',
            count, good_metrics('single'), good_metrics('triple'),
        )
        assert status == expected_status
    state.update(status='continue', streak=0, new_updates=V7_MAX_UPDATES - 10)
    _, count = plan_block(state)
    assert count == 10
