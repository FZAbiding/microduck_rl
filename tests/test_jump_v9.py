"""Physical and manager-order regressions for the failures found in V8."""
import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.jump_control import JumpController, jump_pose_metrics
from mjlab_microduck.jump_v9 import (
    configure_v9, set_learning_rate, validate_metrics, REQUIRED_METRICS, update_best,
    worker_arguments, blocked_jump_experience,
)


def sample(c, step, **overrides):
    values = dict(z=torch.full((c.n,), .115), quat=torch.tensor([[1., 0., 0., 0.]]).repeat(c.n, 1),
        vz=torch.zeros(c.n), omega_xy=torch.zeros(c.n, 3),
        feet=torch.ones(c.n, 2, dtype=torch.bool), body=torch.zeros(c.n, dtype=torch.bool),
        force=torch.zeros(c.n), joint_l1=torch.zeros(c.n), horizontal_speed=torch.zeros(c.n),
        joint_max=torch.zeros(c.n), ankle_max=torch.zeros(c.n),
        foot_pitch_max=torch.zeros(c.n), foot_roll_max=torch.zeros(c.n))
    values.update(overrides)
    c.update(step, **values)


def test_recovery_completes_without_fake_jump_height_or_events():
    c = JumpController(1, policy_version=9)
    c.start_recovery(torch.tensor([0]), torch.zeros(1), torch.zeros(1, 2))
    recovery_events = jump_events = height_mass = 0
    for step in range(450):
        sample(c, step)
        recovery_events += c.recovery_complete_event.item()
        jump_events += c.complete_event.item()
        height_mass += c.height_progress.item()
    assert recovery_events == 1 and jump_events == height_mass == 0
    assert not c.recovery_timeout.item() and not c.busy.item()
    assert c.peak.item() == 0 and c.settle_hold.item()
    assert not c.press(torch.tensor([True])).item()


def test_rejection_survives_command_after_reward_order_exactly_once():
    c = JumpController(1, policy_version=9)
    sample(c, 0)
    c.mark_readiness_failure(torch.tensor([True]))
    sample(c, 1)
    assert c.readiness_failure_event.item() == 1
    sample(c, 1)  # repeated reward calls must see the same event
    assert c.readiness_failure_event.item() == 1
    sample(c, 2)
    assert c.readiness_failure_event.item() == 0
    c.mark_readiness_failure(torch.tensor([True]))
    c.reset(torch.tensor([0]))
    sample(c, 3)
    assert c.readiness_failure_event.item() == 0


def test_no_late_completion_payment_after_recovery_deadline():
    c = JumpController(1, policy_version=9)
    c.start_recovery(torch.tensor([0]), torch.zeros(1), torch.zeros(1, 2))
    for step in range(450):
        sample(c, step, joint_max=torch.tensor([.11]))
    assert c.recovery_timeout.item()
    for step in range(450, 750):
        sample(c, step)
        assert c.recovery_complete_event.item() == c.complete_event.item() == 0
    assert not c.recovery_complete.item() and not c.complete.item()


def test_strict_first_request_and_missing_measurements():
    c = JumpController(1, policy_version=9)
    for step in range(110):
        sample(c, step, ankle_max=torch.tensor([.061]))
    assert not c.press(torch.tensor([True])).item()
    with pytest.raises(ValueError, match='all physical'):
        sample(c, 111, joint_max=None)


@pytest.mark.parametrize('backlash', [False, True])
def test_actual_model_named_ankles_ignore_head_and_passive_joints(backlash):
    import mujoco
    from mjlab_microduck.robot.microduck_constants import get_standup_spec, get_backlash_spec
    model = (get_backlash_spec if backlash else get_standup_spec)().compile()
    names = [model.joint(j).name for j in range(model.njnt)
             if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE and not model.joint(j).name.startswith('passive_')]
    pos = torch.zeros(1, len(names))
    default = pos.clone()
    feet = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]])
    pos[0, names.index('head_pitch')] = 1.
    pos[0, names.index('right_hip_yaw')] = .5
    m = jump_pose_metrics(names, pos, default, feet)
    assert m['joint_max'].item() == .5 and m['ankle_max'].item() == 0
    pos[0, names.index('right_ankle')] = .7
    m = jump_pose_metrics(names, pos, default, feet)
    assert m['ankle_max'].item() == pytest.approx(.7)


def test_foot_cost_uses_controller_feet_not_unresolved_all_sites(monkeypatch):
    from mjlab_microduck.tasks import mdp
    c = JumpController(1, policy_version=9)
    c.settle_hold[:] = True
    monkeypatch.setattr(mdp, 'jump_v2_update', lambda env: c)
    # No scene: if a new unresolved selector touches any other site this fails.
    env = SimpleNamespace()
    assert mdp.jump_foot_pose_cost(env).item() == 0
    c.foot_pitch_max[:] = math.pi/2
    assert mdp.jump_foot_pose_cost(env).item() == 4


def test_v9_measurements_select_two_named_sites(monkeypatch):
    from mjlab_microduck.tasks import mdp
    names = [f'{s}_{j}' for s in ('left','right') for j in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')]
    names += ['head_yaw','head_roll','head_pitch','neck_pitch']
    q = torch.tensor([[[0.,1.,0.,0.], [1.,0.,0.,0.], [1.,0.,0.,0.]]])
    asset = SimpleNamespace(joint_names=names, site_names=['other','left_foot','right_foot'],
        data=SimpleNamespace(joint_pos=torch.zeros(1,14), default_joint_pos=torch.zeros(1,14),
                             site_quat_w=q, root_link_lin_vel_w=torch.zeros(1,3)))
    env = SimpleNamespace(scene={'robot': asset})
    monkeypatch.setattr(mdp, '_servo_joint_ids', lambda env, a: list(range(14)))
    assert mdp.jump_v9_measurements(env)['foot_roll_max'].item() == 0


def test_touchdown_event_has_dt_independent_mass(monkeypatch):
    from mjlab_microduck.tasks import mdp
    c = JumpController(1, policy_version=9)
    c.impact[:] = 1
    env = SimpleNamespace(step_dt=.02, scene={'robot': SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=torch.tensor([[.02,0.,.115]])))})
    monkeypatch.setattr(mdp, 'jump_v2_update', lambda env: c)
    assert (mdp.jump_touchdown_displacement_cost(env)*env.step_dt).item() == pytest.approx(.8)


def test_reset_classification_reused_by_command_reset(monkeypatch):
    from mjlab_microduck.tasks import mdp
    cfg = mdp.JumpRequestCommandCfg(resampling_time_range=(1000.,1000.), policy_version=9,
                                  recovery_probability=.5, standing_probability=.5)
    n = 128
    env = SimpleNamespace(num_envs=n, device='cpu', common_step_counter=0,
        sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(n,21))),
        scene=SimpleNamespace(terrain=SimpleNamespace(env_origins=torch.zeros(n,3))))
    env.sim.data.qpos[:,3] = 1
    c = JumpController(n, policy_version=9)
    env._jump_controller = c
    command = mdp.JumpRequestCommand(cfg, env)
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    spawned = []
    monkeypatch.setattr(mdp, 'set_random_jump_recovery_state', lambda env, ids, **kw: spawned.extend(ids.tolist()))
    ids = torch.arange(n)
    mdp.jump_v9_reset(env, ids)
    expected = command.recovery_only.clone()
    command.reset(ids)
    assert 0 < int(expected.sum()) < n
    assert torch.equal(expected, command.recovery_only)
    assert torch.equal(expected, c.recovery_only)
    assert sorted(spawned) == expected.nonzero().flatten().tolist()
    assert command.request_count.sum() == 0
    # Partial reset must not change classifications outside its subset.
    keep = command.recovery_only[10:].clone()
    mdp.jump_v9_reset(env, ids[:10])
    command.reset(ids[:10])
    assert torch.equal(command.recovery_only[10:], keep)


def valid_metrics(success=.5):
    result = {key: 0. for key in REQUIRED_METRICS}
    result.update(evaluator_contract=9, success_rate=success, sequence_success_rate=success,
                  request_acceptance_rate=1., final_drift_p90_m=.02)
    return result


def test_missing_diagnostics_not_promoted_and_unsafe_cannot_replace_best():
    m = valid_metrics()
    del m['foot_pose_p90_deg']
    with pytest.raises(ValueError, match='missing'):
        validate_metrics(m)
    state = dict(learning_rate=5e-5)
    update_best(state, 'good.pt', valid_metrics(.6), valid_metrics(.6))
    bad = valid_metrics(1.)
    bad['failure_rate'] = 1.
    assert update_best(state, 'bad.pt', bad, bad) == 'rollback'
    assert state['checkpoint'] == state['best_checkpoint'] == 'good.pt'


def test_lr_applied_to_loaded_optimizer():
    parameter = torch.nn.Parameter(torch.zeros(1))
    alg = SimpleNamespace(learning_rate=1e-4, optimizer=torch.optim.Adam([parameter], lr=1e-4))
    set_learning_rate(alg, 5e-5)
    assert alg.learning_rate == alg.optimizer.param_groups[0]['lr'] == 5e-5


def test_worker_follows_only_rollback_pointer_and_hash(tmp_path):
    from mjlab_microduck.jump_artifacts import sha256
    best, rejected = tmp_path/'best.pt', tmp_path/'rejected.pt'
    best.write_bytes(b'best')
    rejected.write_bytes(b'rejected')
    state = dict(checkpoint=str(rejected), best_checkpoint=str(best), learning_rate=5e-5,
                 best_key=[.8, .8, -.02])
    bad = valid_metrics(.2)
    assert update_best(state, str(rejected), bad, bad) == 'rollback'
    args = worker_arguments(state, tmp_path/'new-block')
    assert args[args.index('--resume')+1] == str(best)
    assert args[args.index('--parent-sha256')+1] == sha256(best)
    assert args[args.index('--learning-rate')+1] == str(2.5e-5)


def test_missing_rate_cannot_hide_as_unobserved_event():
    values = valid_metrics()
    values['failure_rate'] = None
    with pytest.raises(ValueError, match='invalid V9 rate'):
        validate_metrics(values)


def test_pilot_stops_when_all_requested_skill_experience_is_blocked():
    progress = dict(measured_steps=1200, reward_mass_per_env=dict(
        jump_readiness_failure=-2.4, jump_height_progress=0., jump_complete=0.,
        jump_recovery_complete=0.))
    assert blocked_jump_experience(progress)
    progress['measured_steps'] = 120  # Smoke ends before the first request.
    assert not blocked_jump_experience(progress)
    progress['measured_steps'] = 1200
    progress['reward_mass_per_env']['jump_height_progress'] = .5
    assert not blocked_jump_experience(progress)


def test_stage_config_and_frozen_legacy():
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg
    from mjlab_microduck.jump_curriculum import configure_v8
    base = make_microduck_jump_env_cfg(nominal_bootstrap=False, command_protocol=2)
    cfg = configure_v9(base)
    assert cfg.commands['twist'].policy_version == 9
    assert cfg.events['jump_reset_state'].func.__name__ == 'jump_v9_reset'
    assert cfg.episode_length_s >= 40
    assert cfg.rewards['jump_settle_foot_pose'].weight == pytest.approx(-.1)
    assert configure_v8(base).rewards['jump_settle_foot_pose'].weight == -.4


def test_idle_exploration_absolute_on_resume_and_preserves_request_distribution():
    from mjlab_microduck.jump_v9 import set_idle_exploration
    from mjlab_microduck.jump_transfer import TransferActor
    from tensordict import TensorDict
    names = [f'{side}_{joint}' for side in ('left', 'right')
             for joint in ('hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle')]
    names += ['neck_pitch', 'head_pitch', 'head_yaw', 'head_roll']
    obs = TensorDict({'actor': torch.zeros(2, 61)}, [2])
    obs['actor'][1, 48] = 1.
    actor = TransferActor(obs, {'actor': ['actor']}, 'actor', 14,
        hidden_dims=(16,), distribution_cfg={'class_name': 'GaussianDistribution',
                                           'init_std': .15, 'std_type': 'scalar'})
    actor.configure_exploration(names)
    actor(obs, stochastic_output=True)
    requested_std = actor.distribution._distribution.stddev[1].clone()
    set_idle_exploration(actor, names, .25)
    saved = actor.idle_std.clone()
    set_idle_exploration(actor, names, .25)
    assert torch.equal(saved, actor.idle_std)
    actor(obs, stochastic_output=True)
    assert torch.allclose(actor.distribution._distribution.stddev[0, :10], torch.full((10,), .0125))
    assert torch.equal(actor.distribution._distribution.stddev[1], requested_std)
    for bad in (0., -1., 2., float('nan')):
        with pytest.raises(ValueError):
            set_idle_exploration(actor, names, bad)
