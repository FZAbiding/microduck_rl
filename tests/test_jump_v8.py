"""Jump-V8 contract tests: strict feet, recovery timeout, and budget ledger."""
import math
import pytest
import torch

from mjlab_microduck.jump_control import JumpController
from mjlab_microduck.jump_curriculum import (
    V8_MAX_UPDATES, V8_REWARD_WEIGHTS, configure_v8, v8_distribution,
    v8_metrics_pass, v8_branch_key,
)
from mjlab_microduck.jump_v8 import (
    initial_state, plan_block, record_smoke, finish_block,
)


def sample(c, step, *, feet=(True, True), z=.115, vz=0., tilt_deg=0.,
           joint_l1=0., joint_max=0., ankle_max=0., foot_pitch=0.,
           foot_roll=0., horizontal_speed=0., yaw_rate=0., body=False):
    a = math.radians(tilt_deg)
    q = torch.tensor([[math.cos(a/2), math.sin(a/2), 0., 0.]])
    c.update(step, torch.tensor([z]), q, torch.tensor([vz]),
             torch.tensor([[0., 0., yaw_rate]]), torch.tensor([feet]),
             torch.tensor([body]), torch.tensor([10.]), torch.tensor([joint_l1]),
             torch.tensor([horizontal_speed]), torch.tensor([joint_max]),
             torch.tensor([ankle_max]), torch.tensor([foot_pitch]),
             torch.tensor([foot_roll]))


def arm():
    c = JumpController(1, dt=.02, stand_z=.115, target_delta=.032,
                       required_delta=.030, policy_version=8)
    for i in range(100):
        sample(c, i)
    assert c.ready_time.item() == pytest.approx(2.)
    assert c.press(torch.tensor([True])).item()
    return c


def takeoff_land(c, start=100):
    sample(c, start, feet=(False, False), z=.15)
    sample(c, start+1, feet=(False, False), z=.15)
    sample(c, start+2, feet=(True, False), z=.12)
    sample(c, start+3)
    assert c.landed.item()
    return start + 4


def good_metrics(sequence=False):
    row = {
        "failure_rate": 0., "body_contact_rate": 0., "invalid_rate": 0.,
        "timeout_rate": 0., "recovery_timeout_rate": 0.,
        "readiness_failure_rate": 0., "request_acceptance_rate": 1.,
        "peak_delta_p10": .03, "final_drift_p90_m": .005,
        "final_drift_max_m": .01, "heading_change_p90_deg": 1.,
        "heading_change_max_deg": 2., "recovery_tilt_p90_deg": 2.,
        "recovery_tilt_max_deg": 3., "foot_pose_p90_deg": 1.,
        "foot_pose_max_deg": 2., "ankle_error_p90_rad": .03,
        "ankle_error_max_rad": .05, "leg_home_max_p90_rad": .05,
        "leg_home_max_rad": .1, "final_horizontal_speed_max_m_s": .02,
        "ready_latency_p90_s": 2., "ready_latency_max_s": 3.,
        "complete_latency_p90_s": 5., "complete_latency_max_s": 6.,
        "busy_stuck_rate": 0., "success_rate": .95,
    }
    if sequence:
        row.update(sequence_success_rate=.95, jump_success_rate=.98)
    return row


def test_v8_distribution_and_fixed_config():
    assert v8_distribution() == {"recovery": .35, "single": .25, "triple": .30, "standing": .10}
    assert v8_distribution(True) == {"recovery": .10, "single": .20, "triple": .60, "standing": .10}
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg
    cfg = configure_v8(make_microduck_jump_env_cfg(nominal_bootstrap=False, command_protocol=2))
    assert cfg.episode_length_s == 32.
    assert cfg.commands["twist"].policy_version == 8
    assert cfg.commands["twist"].target_delta == pytest.approx(.032)
    assert cfg.commands["twist"].required_delta == pytest.approx(.030)
    assert all(cfg.rewards[k].weight == pytest.approx(v) for k, v in V8_REWARD_WEIGHTS.items())
    assert not cfg.curriculum


def test_v8_ready_gate_catches_a_single_joint_and_heel_lift():
    c = JumpController(1, policy_version=8)
    c.accepted[:] = True
    c.ready_time[:] = 2.
    for step, (state_key, sample_key, value) in enumerate(
        (("joint_max", "joint_max", .101),
         ("ankle_max", "ankle_max", .061),
         ("foot_pitch_max", "foot_pitch", math.radians(3.01)),
         ("foot_roll_max", "foot_roll", math.radians(3.01)))
    ):
        c.ready_time[:] = 2.
        setattr(c, state_key, torch.tensor([value]))
        sample(c, step, **{sample_key: value})
        assert c.ready_time.item() == 0.
        assert not c.press(torch.tensor([True])).item()
        setattr(c, state_key, torch.zeros(1))
    c.ready_time[:] = 2.
    assert c.press(torch.tensor([True])).item()


def test_v8_timeout_releases_busy_and_settle_hold_survives_completion():
    c = arm()
    step = takeoff_land(c)
    for i in range(step, step + 250):
        sample(c, i)
    assert c.complete.item() and c.settle_hold.item() and not c.busy.item()
    assert c.phase_mask("settle_hold").item()
    c = arm()
    step = takeoff_land(c)
    events = 0
    for i in range(step, step + 401):
        sample(c, i, tilt_deg=10.)
        events += int(c.recovery_timeout_event.item())
    assert c.recovery_timeout.item() and events == 1
    assert not c.busy.item()


def test_v8_request_rejection_is_an_event_but_not_a_permanent_busy_state():
    c = JumpController(1, policy_version=8)
    c.ready_time[:] = 2.
    c.busy[:] = True
    c.mark_readiness_failure(torch.tensor([True]))
    assert c.readiness_failure.item() and c.readiness_failure_event.item() == 1.
    c.mark_readiness_failure(torch.tensor([True]))
    assert c.readiness_failure_event.item() == 1.


def test_v8_metrics_and_two_branch_ledger():
    assert v8_metrics_pass(good_metrics(), good_metrics(True))
    state = initial_state("a.pt", "b.pt")
    record_smoke(state)
    assert state["new_updates"] == 5
    for branch in ("A", "A", "B", "B"):
        name, count = plan_block(state)
        assert name == branch and count == 250
        finish_block(state, branch, branch + str(count), count,
                     good_metrics(), good_metrics(True))
    assert state["selected_branch"] in ("A", "B")
    assert state["new_updates"] == 1005
    state["status"] = "continue"
    assert plan_block(state)[1] == 250
    state["new_updates"] = V8_MAX_UPDATES
    assert plan_block(state) == (None, 0)
