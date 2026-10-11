"""Physical state-bank contracts and origin-safe partial recovery resets."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab_microduck.jump_landing_bank import load_bank, save_bank


def make_bank(path):
    row = dict(qpos=np.array([1., 2., .12, 1., 0., 0., 0., .103]),
               qvel=np.arange(7, dtype=float) * .001, joint_max=np.asarray(.103),
               episode_id=np.asarray(3), offset_s=np.asarray(2.))
    save_bank(path, [row], ['left_knee'], dict(checkpoint_sha256='parent', seed=42))
    return row


def test_bank_rejects_wrong_model_and_invalid_physics(tmp_path):
    path = tmp_path/'bank.npz'
    make_bank(path)
    arrays, meta = load_bank(path, ['left_knee'], 8, 7)
    assert meta['seed'] == 42 and arrays['qpos'].shape == (1, 8)
    with pytest.raises(ValueError, match='contract mismatch'):
        load_bank(path, ['right_knee'], 8, 7)
    with pytest.raises(ValueError, match='dimensions mismatch'):
        load_bank(path, ['left_knee'], 9, 7)
    with np.load(path) as data:
        contents = {k: data[k].copy() for k in data.files}
    contents['qpos'][0, 3] = 2.
    np.savez(path, **contents)
    with pytest.raises(ValueError, match='quaternion'):
        load_bank(path, ['left_knee'], 8, 7)
    contents['qpos'][0, 3] = float('nan')
    np.savez(path, **contents)
    with pytest.raises(ValueError, match='nonfinite'):
        load_bank(path, ['left_knee'], 8, 7)


def test_partial_reset_origins_do_not_accumulate(tmp_path):
    from mjlab_microduck.tasks import mdp
    path = tmp_path/'bank.npz'
    row = make_bank(path)
    class Scene(dict):
        terrain = SimpleNamespace(env_origins=torch.tensor([[10., 20., 0.], [30., 40., 1.]]))
    env = SimpleNamespace(device='cpu', scene=Scene(robot=SimpleNamespace(joint_names=['left_knee'])),
        sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(2, 8, dtype=torch.float64),
                                                 qvel=torch.zeros(2, 7, dtype=torch.float64))))
    ids = torch.tensor([1])
    mdp.set_jump_landing_bank_state(env, ids, path)
    first = env.sim.data.qpos.clone()
    assert torch.equal(first[0], torch.zeros(8, dtype=torch.float64))
    assert torch.allclose(first[1, :3], torch.tensor([31., 42., 1.12], dtype=torch.float64))
    mdp.set_jump_landing_bank_state(env, ids, path)
    assert torch.equal(first, env.sim.data.qpos)
    assert torch.equal(env.sim.data.qvel[1], torch.from_numpy(row['qvel']))
    with pytest.raises(ValueError, match='no samples'):
        mdp.set_jump_landing_bank_state(env, ids, path, max_joint_error=.10)
    mdp.set_jump_landing_bank_state(env, ids, path, max_joint_error=.11)
    assert torch.equal(first, env.sim.data.qpos)


def test_landing_curriculum_preserves_pilot_quality_and_requires_new_reset_distribution():
    from mjlab_microduck.jump_v9 import configure_v9
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg
    def cfg(stage):
        return configure_v9(make_microduck_jump_env_cfg(nominal_bootstrap=False, command_protocol=2), stage)
    pilot, recovery = cfg('pilot'), cfg('landing_recovery')
    command = recovery.commands['twist']
    assert (command.standing_probability, command.recovery_probability,
            command.single_probability, command.triple_probability) == (.30, .35, .35, .0)
    for key in pilot.rewards:
        assert pilot.rewards[key].weight == recovery.rewards[key].weight
    assert command.strict_first_request


def test_action_and_yaw_history_restored_once_after_manager_reset(tmp_path):
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.jump_control import JumpController
    path = tmp_path/'history.npz'
    row = dict(qpos=np.array([0., 0., .12, 1., 0., 0., 0.]), qvel=np.zeros(6),
               joint_max=np.asarray(.103), episode_id=np.asarray(1), offset_s=np.asarray(5.),
               action=np.full(14, .1), prev_action=np.full(14, .2),
               prev_prev_action=np.full(14, .3), target_yaw=np.asarray(.4), heading_error=np.asarray(-.1))
    save_bank(path, [row], [], dict(seed=42))
    class Scene(dict):
        terrain = SimpleNamespace(env_origins=torch.zeros(2, 3))
    manager = SimpleNamespace(**{'_'+k:torch.zeros(2,14) for k in ('action','prev_action','prev_prev_action')})
    env = SimpleNamespace(device='cpu', num_envs=2, scene=Scene(robot=SimpleNamespace(joint_names=[])),
        sim=SimpleNamespace(data=SimpleNamespace(qpos=torch.zeros(2,7,dtype=torch.float64),
                                                 qvel=torch.zeros(2,6,dtype=torch.float64))),
        action_manager=manager, _jump_controller=JumpController(2,policy_version=9))
    mdp.set_jump_landing_bank_state(env, torch.tensor([1]), path)
    mdp.restore_jump_landing_history(env, torch.tensor([0,1]))
    assert torch.allclose(manager._action[1],torch.full((14,),.1))
    assert torch.all(manager._action[0] == 0)
    assert env._jump_controller.target_yaw[1] == pytest.approx(.4)
    assert env._jump_controller.heading_error[1] == pytest.approx(-.1)
    manager._action[1] = .9
    mdp.restore_jump_landing_history(env, torch.tensor([1]))
    assert torch.allclose(manager._action[1],torch.full((14,),.9))


def test_height_shortfall_is_not_misclassified_as_pose_recovery_failure():
    from mjlab_microduck.jump_v9_evaluation import height_failure_breakdown
    result = height_failure_breakdown([
        dict(requested_jumps=1, completed_jumps=0, peak_deltas_m=[.029]),
        dict(requested_jumps=1, completed_jumps=0, peak_deltas_m=[.031]),
        dict(requested_jumps=1, completed_jumps=1, peak_deltas_m=[.032])])
    assert result['height_shortfall_requests'] == 1
    assert result['height_qualified_uncompleted_requests'] == 1
    assert result['requested_height_shortfall_rate'] == pytest.approx(1/3)
    assert height_failure_breakdown([])['requested_height_shortfall_rate'] is None
    boundary = height_failure_breakdown([dict(requested_jumps=1, completed_jumps=1,
                                             peak_deltas_m=[.030 - .5e-6])])
    assert boundary['height_qualified_uncompleted_requests'] == 0
