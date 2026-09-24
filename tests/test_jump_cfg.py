"""Jump transfer contract and adversarial state-machine regression tests (CPU)."""
import copy
from pathlib import Path
import pytest
import torch
from mjlab_microduck.jump_control import JumpController


def sample(c, step, feet=(True, True), z=.115, vz=0., body=False, tilt=0.):
    import math
    q=torch.tensor([[math.cos(tilt/2), math.sin(tilt/2), 0., 0.]]).repeat(c.n,1)
    c.update(step, torch.full((c.n,),z), q, torch.full((c.n,),vz),
             torch.zeros(c.n,2), torch.tensor([feet]).repeat(c.n,1),
             torch.full((c.n,),body), torch.ones(c.n)*10)


def armed(n=1):
    c=JumpController(n)
    for step in range(50):
        sample(c,step)
    assert c.press(torch.ones(n,dtype=torch.bool)).all()
    return c


def test_jump_task_contract():
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab_microduck.jump_runner import JumpRunner
    for task in ('Mjlab-Jump-Flat-MicroDuck','Mjlab-Jump-Flat-Backlash-MicroDuck'):
        cfg=load_env_cfg(task)
        assert cfg.episode_length_s == 6
        assert not cfg.curriculum
        assert cfg.commands['twist'].standing_probability == .25
        assert cfg.observations['actor'].terms['head_command'].params['dim'] == 4
        assert cfg.observations['actor'].terms['body_command'].params['dim'] == 6
        assert cfg.events['expand_bam_friction_fields'].mode == 'startup'
        assert cfg.rewards['jump_launch_velocity'].weight > 0
        assert 'jump_launch_drive' not in cfg.rewards
        assert cfg.rewards['jump_launch_progress'].weight == 1
        assert cfg.rewards['jump_landing_impact'].weight < 0
        assert cfg.rewards['body_impact'].weight < 0
        assert cfg.rewards['action_rate_l2'].weight == 0
        assert load_runner_cls(task) is JumpRunner
        rl=load_rl_cfg(task)
        assert rl.algorithm.learning_rate == 1e-4
        assert rl.actor.distribution_cfg['init_std'] == .15
        assert rl.algorithm.symmetry_cfg is None
        model=cfg.scene.entities['robot'].spec_fn().compile()
        ids=[j for j in range(model.njnt) if model.jnt_type[j] != 0 and not model.joint(j).name.startswith('passive_')]
        assert len(ids)==14
        # Excluded foot bodies contain no additional collidable non-sole geoms.
        for i in range(model.ngeom):
            if model.geom_contype[i] and model.body(int(model.geom_bodyid[i])).name in ('ankle_left','ankle_right'):
                assert model.geom(i).name in ('left_foot_collision','right_foot_collision')


def test_request_requires_support_and_rejects_repeated_press():
    c=JumpController(1)
    assert not c.press(torch.tensor([True])).item()
    for i in range(100):
        sample(c,i,feet=(False,False),z=.15)
    assert not c.took_off.item()
    c=armed()
    assert not c.press(torch.tensor([True])).item()
    assert c.request.item()


def test_single_foot_never_counts_as_takeoff_and_timeout_clears_request():
    c=armed()
    for i in range(50,126):
        sample(c,i,feet=(False,True),z=.16)
    assert not c.took_off.item()
    assert c.timeout.item() and not c.request.item()
    assert c.height_progress.item()==0
    # A late flight after the request expired cannot retroactively qualify.
    sample(c,126,feet=(False,False),z=.16)
    sample(c,127,feet=(False,False),z=.16)
    assert not c.took_off.item()


def test_launch_progress_cannot_be_farmed_by_repeated_crouches():
    c = armed()
    total = 0.
    for i in range(50,120):
        sample(c,i,vz=.8 if i%2 else 0.)
        total += c.launch_progress.item()
    assert total == pytest.approx(.1)
    sample(c,120,feet=(False,False),z=.12,vz=1.)
    assert c.launch_progress.item()==0


def test_real_peak_is_uncapped_rewards_only_new_flight_progress():
    c=armed()
    sample(c,50,feet=(False,False),z=.14)
    assert not c.took_off.item()
    sample(c,51,feet=(False,False),z=.14)
    assert c.took_off.item() and not c.request.item()
    assert c.height_progress.item()==pytest.approx(1.)
    sample(c,52,feet=(False,False),z=.20)
    assert c.peak.item()==pytest.approx(.20)
    assert c.height_progress.item()==0
    sample(c,53,feet=(False,False),z=.15)
    assert c.height_progress.item()==0


def test_first_touchdown_impact_and_success_paid_once_full_follow_through():
    c=armed()
    sample(c,50,feet=(False,False),z=.14,vz=.2)
    sample(c,51,feet=(False,False),z=.14,vz=-.3)
    sample(c,52,feet=(True,False),vz=0.)
    assert c.impact.item()>0 and not c.landed.item()
    sample(c,53,feet=(False,False),z=.13)
    sample(c,54)
    assert c.landed.item() and c.impact.item()==0
    payments=0
    for i in range(55,117):
        sample(c,i)
        payments+=c.success_event.item()
        if i<115:
            assert not c.complete.item()
    assert payments==1 and c.complete.item()
    assert c.press(torch.tensor([True])).item()
    assert not c.took_off.item() and not c.complete.item()


def test_body_contact_or_nan_invalidates_positive_rewards():
    for body,z in ((True,.15),(False,float('nan'))):
        c=armed()
        sample(c,50,feet=(False,False),z=z,body=body)
        sample(c,51,feet=(False,False),z=.15)
        assert c.invalid.item()
        assert c.height_progress.item()==0
        assert not c.complete.item()


def test_partial_reset_and_same_step_updates_do_not_read_stale_sensors():
    c=armed(2)
    sample(c,50,feet=(False,False),z=.14)
    before=c.air_steps.clone()
    sample(c,50,feet=(False,False),z=.20)
    assert torch.equal(before,c.air_steps)
    c.reset(torch.tensor([0]),step=50)
    sample(c,50,feet=(False,False),z=.20,body=True)
    assert not c.invalid[0] and c.peak[0]==0
    sample(c,51,feet=(False,False),z=.14)
    assert not c.took_off[0] and c.took_off[1]
    assert c.height_progress[0]==0


def test_frozen_normalization_survives_train_mode_and_new_request_has_gradient():
    from tensordict import TensorDict
    from mjlab_microduck.jump_transfer import TransferActor
    obs=TensorDict({'actor':torch.randn(8,61)},[8])
    actor=TransferActor(obs,{'actor':['actor']},'actor',14,obs_normalization=True)
    before=copy.deepcopy(actor.obs_normalizer.state_dict())
    actor.train(); actor.update_normalization(obs)
    for k,v in before.items():
        assert torch.equal(v,actor.obs_normalizer.state_dict()[k])
    actor.mlp[0].weight.data[:,48]=0
    obs['actor'][:,48]=1
    actor(obs).sum().backward()
    assert actor.mlp[0].weight.grad[:,48].abs().sum()>0


def test_official_import_matches_onnx_and_preserves_zero_command(tmp_path):
    from mjlab_microduck.jump_transfer import prepare_transfer
    path=Path('artifacts/official/alpha_stand.onnx')
    if not path.exists():
        pytest.skip('Official ONNX not bundled in git')
    result=prepare_transfer(path,tmp_path/'standing_init.pt')
    assert result['onnx_max_error']<5e-5
    assert result['zero_command_max_error']<5e-5


def test_small_jump_curriculum_gate_can_advance_without_dr_or_landing():
    from mjlab_microduck.jump_curriculum import record_evaluation
    state = {'stage':0,'streak':0,'evaluations':[]}
    nominal = {'success_rate':0.,'height_rate':.8,'takeoff_rate':1.,'peak_delta_p90':.02}
    standing = {'success_rate':.95}
    for i in (250,500):
        assert record_evaluation(state,i,nominal,standing=standing)=='continue'
    assert record_evaluation(state,750,nominal,standing=standing)=='advance'
    assert state['stage']==1


def test_fixed_episode_ledger_early_failure_late_success():
    from mjlab_microduck.jump_evaluation import EpisodeLedger
    for order in ((0,1),(1,0)):
        ledger=EpisodeLedger(2)
        for idx in order:
            assert ledger.add(idx,{'success':bool(idx)})
            assert not ledger.add(idx,{'success':not bool(idx)})
        assert ledger.complete
        assert sum(r['success'] for r in ledger.records)/2==.5


def test_small_target_has_no_five_mm_discount_and_invalid_peak_cannot_grow():
    c=armed()
    sample(c,50,feet=(False,False),z=.126)
    sample(c,51,feet=(False,False),z=.126)
    for i in range(52,120): sample(c,i)
    assert not c.complete.item()
    c=armed()
    sample(c,50,feet=(False,False),z=.14)
    sample(c,51,feet=(False,False),z=.14)
    sample(c,52,feet=(False,False),z=.5,body=True)
    assert c.peak.item()==pytest.approx(.14)


def test_event_return_dt_scaling_and_no_duplicate_payments():
    from types import SimpleNamespace
    from mjlab_microduck.tasks.mdp import jump_v2_reward
    c=armed()
    sample(c,50,feet=(False,False),z=.15)
    sample(c,51,feet=(False,False),z=.15)
    # Real reward manager multiplication by dt must recover the event return.
    import unittest.mock as mock
    with mock.patch('mjlab_microduck.tasks.mdp.jump_v2_update',return_value=c):
        assert (jump_v2_reward(SimpleNamespace(step_dt=.02),'launch_signal')*.02).item()==pytest.approx(1.)
    total={'launch_signal':0.,'height_progress':0.,'success_event':0.,'complete_event':0.}
    for i in range(51,125):
        if i>51: sample(c,i)
        for k in total: total[k]+=c.state[k].item()
    assert total==pytest.approx(dict(launch_signal=1,height_progress=1,success_event=1,complete_event=1))


def test_conditional_exploration_logprob_recomputed_from_same_observation():
    from tensordict import TensorDict
    from mjlab_microduck.jump_transfer import TransferActor
    names=['hip_yaw_left','hip_roll_left','hip_pitch_left','knee_left','ankle_left',
           'neck_pitch','head_pitch','head_yaw','head_roll',
           'hip_yaw_right','hip_roll_right','hip_pitch_right','knee_right','ankle_right']
    obs=TensorDict({'actor':torch.randn(12,61)},[12]);obs['actor'][:,48]=torch.arange(12)%2
    for branch in ('A','B'):
        actor=TransferActor(obs,{'actor':['actor']},'actor',14,
            distribution_cfg={'class_name':'GaussianDistribution','init_std':.15})
        actor.configure_exploration(names,branch)
        action=actor(obs,stochastic_output=True)
        old_log=actor.get_output_log_prob(action).clone()
        old=tuple(t.clone() for t in actor.output_distribution_params)
        actor(obs,stochastic_output=True)
        assert torch.equal(old_log,actor.get_output_log_prob(action))
        assert actor.get_kl_divergence(old,actor.output_distribution_params).abs().max()==0
        assert actor.output_std[0,5].item()==pytest.approx(.03)
        assert actor.output_std[1,2].item()==pytest.approx(.2 if branch=='B' else .15)


def test_request_timeout_is_local_but_episode_failure_is_not_cleared():
    c=armed()
    for i in range(50,126):sample(c,i)
    assert c.timeout.item() and c.request_timeout.item()
    assert c.press(torch.tensor([True])).item()
    assert c.timeout.item() and not c.request_timeout.item()
    assert c.paid_launch.item()==0 and c.paid_height.item()==0


def test_two_consecutive_safe_air_samples_are_required():
    import math
    c=armed()
    sample(c,50,feet=(False,False),z=.2,tilt=math.radians(50))
    sample(c,51,feet=(False,False),z=.14)
    assert not c.took_off.item()
    sample(c,52,feet=(False,False),z=.14)
    assert c.took_off.item()


def test_dr_ramps_preserve_event_modes_nominal_centers_and_integer_delays():
    from mjlab_microduck.jump_curriculum import configure_stage
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg
    base=make_microduck_jump_env_cfg(nominal_bootstrap=False)
    for stage,strength in ((3,.25),(4,.5),(5,1.)):
        c=configure_stage(base,stage)
        assert {k:v.mode for k,v in c.events.items()}=={k:v.mode for k,v in base.events.items()}
        assert c.events['foot_friction'].params['ranges']==pytest.approx((1-.3*strength,1+.3*strength))
        for a in c.scene.entities['robot'].articulation.actuators:
            assert isinstance(a.delay_min_lag,int) and isinstance(a.delay_max_lag,int)
            assert a.vin_range==pytest.approx((7.4+strength*(6.5-7.4),7.4+strength*(8.2-7.4)))
        assert c.events['push_robot'].params['velocity_range']['x']==((-0.08,.08) if stage==5 else (-0.,0.))
    assert base.events['foot_friction'].params['ranges']==(.7,1.3)


def test_smoothness_rolls_back_when_takeoff_drops_over_five_points():
    from mjlab_microduck.jump_curriculum import record_evaluation
    state={'stage':2,'streak':0,'evaluations':[],'action_weight':-.02,
           'smoothness_previous':0.,'smoothness_takeoff':.9}
    nominal={'success_rate':.8,'height_rate':.8,'takeoff_rate':.84,'peak_delta_p90':.03}
    assert record_evaluation(state,1000,nominal,standing={'success_rate':1.})=='rollback_smoothness'
    assert state['action_weight']==0 and state['streak']==0



def test_v4_heading_wrap_lock_and_repeated_request_is_stable():
    import math
    from mjlab_microduck.jump_control import encode_heading_error, wrap_to_pi
    c = JumpController(1)
    for step in range(50):
        sample(c, step)
    target = torch.tensor([math.pi - .01])
    assert c.press(torch.tensor([True]), target).item()
    locked = c.target_yaw.clone()
    assert not c.press(torch.tensor([True]), torch.tensor([-.5])).item()
    assert torch.equal(c.target_yaw, locked)
    # Current yaw just across -pi must produce the short -0.02 rad path.
    yaw = -math.pi + .01
    q = torch.tensor([[math.cos(yaw/2), 0., 0., math.sin(yaw/2)]])
    c.update(50, torch.tensor([.115]), q, torch.tensor([0.]),
             torch.tensor([[0., 0., 0.]]), torch.tensor([[True, True]]),
             torch.tensor([False]), torch.tensor([0.]))
    assert c.heading_error.item() == pytest.approx(-.02, abs=2e-4)
    assert c.max_heading_error.item() == pytest.approx(.02, abs=2e-4)
    assert encode_heading_error(c.heading_error).abs().item() <= .1
    assert wrap_to_pi(torch.tensor([math.pi + .2])).item() == pytest.approx(-math.pi + .2, abs=1e-6)
    c.reset(torch.tensor([0]), step=50)
    assert c.target_yaw.item() == 0 and c.heading_error.item() == 0 and not c.accepted.item()


def test_v4_protocol_and_acceptance_contract():
    from mjlab_microduck.jump_control import command_protocol
    from mjlab_microduck.jump_curriculum import configure_v4
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg
    assert command_protocol({}) == 1
    assert command_protocol({'jump_command_protocol': '2'}) == 2
    base = make_microduck_jump_env_cfg(nominal_bootstrap=False, command_protocol=2)
    cfg = configure_v4(base, target_delta=.04, heading_weight=-.2,
                       yaw_rate_weight=-.02, head_bias_weight=1.0)
    assert cfg.commands['twist'].protocol == 2
    assert cfg.rewards['action_rate_l2'].weight == pytest.approx(-.02)
    assert cfg.rewards['jump_heading_error'].weight == pytest.approx(-.2)
    assert cfg.rewards['jump_yaw_rate'].weight == pytest.approx(-.02)
    assert cfg.rewards['jump_head_bias'].weight == pytest.approx(1.0)
    c = JumpController(1, target_delta=.04)
    assert c.acceptance_delta == pytest.approx(.035)
    c.target_delta = .035
    assert c.acceptance_delta == pytest.approx(.030)
    c.target_delta = .02
    assert c.acceptance_delta == pytest.approx(.020)
    # Protocol 1 retains a zero legacy third command slot.
    c.heading_error[:] = .5
    assert c.command(1)[:, 2].item() == 0
    assert c.command(2)[:, 2].item() == pytest.approx(.1)
