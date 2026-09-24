"""Evaluation uses training cfgs, explicit randomization and raw terminal states."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
from pathlib import Path
import torch
import math


def nominalize(cfg):
    # Whitelist deterministic initialization. A newly added randomizer cannot
    # silently leak into nominal evaluation.
    keep = {'reset_base', 'reset_robot_joints', 'expand_bam_friction_fields',
            'reset_action_history', 'jump_reset_state'}
    cfg.events = {k: v for k, v in cfg.events.items() if k in keep}
    cfg.events['reset_base'].params['pose_range'].update({k: (0., 0.) for k in ('x', 'y', 'yaw', 'roll', 'pitch')})
    cfg.events['reset_base'].params['velocity_range'] = {k: (0., 0.) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
    cfg.events['reset_robot_joints'].params['position_range'] = (-.01, .01)
    cfg.events['reset_robot_joints'].params['velocity_range'] = (0., 0.)
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values():
            term.delay_min_lag = term.delay_max_lag = 0
            if 'max_angle_deg' in term.params:
                term.params['max_angle_deg'] = 0.
            if 'biased' in term.params:
                term.params['biased'] = False
    from copy import deepcopy
    cfg.scene.entities = deepcopy(cfg.scene.entities)
    for actuator in cfg.scene.entities['robot'].articulation.actuators:
        for name, value in (('delay_min_lag', 0), ('delay_max_lag', 0), ('vin_range', (7.4, 7.4)),
                            ('vin_drop_gain_range', (.1, .1))):
            if hasattr(actuator, name):
                setattr(actuator, name, value)
    cfg.curriculum.clear()


def make_eval_env(episodes=100, mode='nominal', standing=False, seed=42, device='cuda:0', video=False, stage=0):
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg
    cfg = make_microduck_jump_env_cfg(play=False, nominal_bootstrap=mode == 'nominal')
    cfg.seed = seed
    cfg.scene.num_envs = episodes
    cfg.auto_reset = False
    cfg.curriculum.clear()
    cfg.episode_length_s = 10. if standing else 6.
    cfg.commands['twist'].standing_probability = 1. if standing else 0.
    if mode == 'nominal':
        nominalize(cfg)
    elif stage:
        # DR evaluation must use the same evidence-gated difficulty slice as
        # training. Keep nominal evaluation deterministic above.
        from mjlab_microduck.jump_curriculum import configure_stage
        configure_stage(cfg, int(stage))
    return ManagerBasedRlEnv(cfg, device=device, render_mode='rgb_array' if video else None)



def _quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def _quat_conj(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def _quat_rpy(q):
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = torch.asin((2*(w*y-z*x)).clamp(-1, 1))
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return torch.stack((roll, pitch, yaw), -1)


def _head_metrics(env):
    """Return measured four-joint errors and head pose relative to trunk."""
    robot = env.scene['robot']
    names = tuple(robot.joint_names)
    ids = [names.index(n) for n in ('neck_pitch', 'head_pitch', 'head_yaw', 'head_roll') if n in names]
    if len(ids) != 4:
        errors = torch.zeros(env.num_envs, 4, device=env.device)
    else:
        errors = robot.data.joint_pos[:, ids] - robot.data.default_joint_pos[:, ids]
        # Backlash encoders observe the output side, matching the policy.
        for j, n in enumerate(('neck_pitch', 'head_pitch', 'head_yaw', 'head_roll')):
            b = names.index('passive_' + n + '_backlash') if 'passive_' + n + '_backlash' in names else None
            if b is not None:
                errors[:, j] += robot.data.joint_pos[:, b]
    body_names = tuple(getattr(robot, 'body_names', ()))
    try:
        trunk = body_names.index('trunk_base')
        head = body_names.index('jaw_soft')
        rel = _quat_mul(_quat_conj(robot.data.body_link_quat_w[:, trunk]),
                        robot.data.body_link_quat_w[:, head])
        rel_rpy = _quat_rpy(rel)
    except (ValueError, AttributeError, IndexError):
        rel_rpy = torch.zeros(env.num_envs, 3, device=env.device)
    return errors, rel_rpy


def _wrap_scalar(angle):
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def raw_sample(env):
    from mjlab_microduck.tasks import mdp
    robot = env.scene['robot']
    return {'z': robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2],
            'quat': robot.data.root_link_quat_w,
            'vz': robot.data.root_link_lin_vel_w[:, 2],
            'omega_xy': robot.data.root_link_ang_vel_b,
            'feet': mdp._jump_feet_contact(env, 'feet_ground_contact'),
            'body': mdp._jump_any_contact(env, 'body_ground_contact')}


def standing_gate(onnx, init_checkpoint, output, device='cuda:0', episodes=100, seed=42):
    import json
    import numpy as np
    import onnxruntime as ort
    from tensordict import TensorDict
    from mjlab_microduck.jump_transfer import import_actor
    env = make_eval_env(episodes, standing=True, seed=seed, device=device)
    teacher, metadata = import_actor(onnx)
    actor = __import__('copy').deepcopy(teacher)
    data = torch.load(init_checkpoint, weights_only=False, map_location='cpu')
    actor.load_state_dict(data['actor_state_dict'])
    actor.to(device)
    session = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'])
    # Validate the observed joint/action contract against ONNX metadata.
    robot = env.scene['robot']
    servo = [n for n in robot.joint_names if not n.startswith('passive_')]
    if servo != metadata['joint_names'].split(','):
        raise ValueError('Joint order mismatch')
    expected_home = np.array(metadata['default_joint_pos'].split(','), dtype=float)
    ids = [robot.joint_names.index(n) for n in servo]
    actual_home = robot.data.default_joint_pos[0, ids].cpu().numpy()
    if not np.allclose(expected_home, actual_home, atol=.00051):
        raise ValueError('HOME mismatch')
    if env.cfg.actions['joint_pos'].scale != float(metadata['action_scale']):
        raise ValueError('Action scale mismatch')
    obs, _ = env.reset()
    failures = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    maxima = torch.zeros(env.num_envs, device=device)
    heights, action_error = [], 0.
    # Half the trajectories are driven by ONNX, half by reconstructed actor.
    ort_mask = torch.arange(env.num_envs, device=device) % 2 == 0
    for step in range(500):
        with torch.inference_mode():
            actions = actor(TensorDict(obs, batch_size=[env.num_envs]))
        x = obs['actor'].cpu().numpy()
        reference = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0] for row in x])
        reference = torch.as_tensor(reference, device=device)
        action_error = max(action_error, float((reference - actions).abs().max()))
        actions = torch.where(ort_mask[:, None], reference, actions)
        obs, reward, terminated, timed_out, _ = env.step(actions)
        s = raw_sample(env)
        tilt = torch.rad2deg(torch.acos((1 - 2 * s['quat'][:, 1:3].square().sum(-1)).clamp(-1, 1)))
        maxima = torch.maximum(maxima, tilt)
        failures |= terminated | s['body'] | (tilt > 15) | ~torch.isfinite(s['z'])
        failures |= ~torch.isfinite(actions).all(-1) | ~torch.isfinite(obs['actor']).all(-1)
        if step >= 150:
            heights.append(s['z'].clone())
    z = torch.stack(heights)
    survivors = ~failures
    result = {'episodes': env.num_envs, 'seconds': 10, 'seed': seed, 'mode': 'nominal',
              'source_sha256': data['source_sha256'],
              'init_sha256': hashlib.sha256(Path(init_checkpoint).read_bytes()).hexdigest(),
              'standing_fraction': float(survivors.float().mean()),
              'onnx_standing_fraction': float(survivors[ort_mask].float().mean()),
              'torch_standing_fraction': float(survivors[~ort_mask].float().mean()),
              'closed_loop_action_max_error': action_error,
              'max_tilt_deg_p90': float(torch.quantile(maxima, .9)),
              'stand_z': float(z[:, survivors].median()) if survivors.any() else None,
              'passed': bool(survivors.float().mean() >= .95 and action_error < 5e-5),
              'note': 'Height uses only surviving closed-loop trajectories after 3 s; no zero-action settle.'}
    env.close()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    return result


class EpisodeLedger:
    """One immutable result per preassigned episode, independent of reset order."""
    def __init__(self, episodes):
        self.records = [None] * episodes

    def add(self, episode_id, record):
        if self.records[episode_id] is None:
            self.records[episode_id] = {'episode_id': episode_id, **record}
            return True
        return False

    @property
    def complete(self):
        return all(r is not None for r in self.records)


def physical_trace(env):
    import mujoco
    robot = env.scene['robot']
    model = env.sim.mj_model
    data = env.sim.data
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot/trunk_base')
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'robot/'+n)
           for n in ('left_foot_collision','right_foot_collision')]
    # These models use box soles. Exact world vertical extent of the box.
    sizes = torch.as_tensor(model.geom_size[ids], device=env.device, dtype=torch.float32)
    rot = data.geom_xmat[:, ids].reshape(env.num_envs, 2, 3, 3)
    clearance = data.geom_xpos[:, ids, 2] - (rot[:,:,2].abs()*sizes).sum(-1)
    origins = env.scene.terrain.env_origins
    return {'com_z': data.subtree_com[:, root, 2] - origins[:,2],
            'foot_clearance':clearance-origins[:,2:3],
            'position':robot.data.root_link_pos_w-origins,
            'joint_vel':robot.data.joint_vel,
            'torque':robot.data.actuator_force}


def evaluate_checkpoint(checkpoint, episodes=100, mode='nominal', seed=123,
                        device='cuda:0', trace_path=None, video_path=None, stage=None,
                        standing=False, dr_strength=None, sequence=None):
    import numpy as np
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg
    from mjlab_microduck.jump_runner import JumpRunner
    from mjlab_microduck.jump_control import JumpController
    from mjlab_microduck.jump_curriculum import configure_stage
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    if episodes <= 0:
        raise ValueError('episodes must be positive')
    before_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    saved = torch.load(checkpoint, weights_only=False, map_location='cpu')
    state = saved['infos']['jump_transfer']
    stage = state['curriculum']['stage'] if stage is None else stage
    protocol = int(state.get('command_protocol', 1))
    base_cfg = make_microduck_jump_env_cfg(nominal_bootstrap=False, command_protocol=protocol)
    checkpoint_format = saved.get('format', 'microduck-jump-v3')
    if checkpoint_format in ('microduck-jump-v7', 'microduck-jump-v8'):
        if mode != 'nominal':
            raise ValueError('Jump V7/V8 foundation evaluation is nominal-only')
        return evaluate_v7_checkpoint(
            checkpoint, episodes, seed, device, trace_path, video_path,
            sequence=sequence or 'single', standing=standing,
        )
    if checkpoint_format == 'microduck-jump-v6':
        from mjlab_microduck.jump_curriculum import configure_v6
        cfg = configure_v6(base_cfg, state.get('target_delta', .03))
    elif checkpoint_format == 'microduck-jump-v5':
        from mjlab_microduck.jump_curriculum import configure_v5
        weights = state.get('reward_weights', {})
        cfg = configure_v5(
            base_cfg, state.get('target_delta', .03),
            weights.get('jump_height_progress', 1.0),
            weights.get('jump_heading_error', -.20),
            weights.get('jump_yaw_rate', -.02),
            weights.get('jump_head_bias', .5),
        )
    elif protocol >= 2:
        from mjlab_microduck.jump_curriculum import configure_v4, v4_posture_schedule
        posture = int(state.get('posture_stage', 1))
        hw, yw, hb = v4_posture_schedule(min(max(posture, 0), 4))
        cfg = configure_v4(base_cfg, state.get('target_delta', .03),
                           state.get('reward_weights', {}).get('jump_heading_error', hw),
                           state.get('reward_weights', {}).get('jump_yaw_rate', yw),
                           state.get('reward_weights', {}).get('jump_head_bias', hb))
        if mode == 'nominal':
            # v4 uses the same deterministic fixed-sample contract as v3;
            # protocol selection must not silently re-enable DR.
            from mjlab_microduck.jump_evaluation import nominalize
            nominalize(cfg)
    else:
        cfg = configure_stage(base_cfg, stage,
            dr_strength=0. if mode=='nominal' else dr_strength,
            action_weight=state['reward_weights']['action_rate_l2'])
    cfg.seed, cfg.scene.num_envs, cfg.auto_reset = seed, episodes, False
    cfg.episode_length_s = 10. if standing else 6.
    # Evaluator schedules requests itself, using the shared controller and raw
    # post-step measurements. No dependency on training manager's stale sample.
    cfg.commands['twist'].standing_probability = 1.
    env = ManagerBasedRlEnv(cfg, device=device, render_mode='rgb_array' if video_path else None)
    vec = RslRlVecEnvWrapper(env)
    runner = JumpRunner(vec, asdict(load_rl_cfg('Mjlab-Jump-Flat-MicroDuck')), device=device)
    runner.load(str(checkpoint), map_location=device)
    policy = runner.get_inference_policy(device=device)
    judge = JumpController(episodes, device, env.step_dt, state['stand_z'],
                           state['target_delta'], state.get('required_delta'))
    judge.protocol = protocol
    rng = np.random.default_rng(seed)
    request_at = torch.as_tensor(rng.uniform(1., 2., episodes), device=device)
    ledger = EpisodeLedger(episodes)
    done_mask = torch.zeros(episodes, dtype=torch.bool, device=device)
    failed = done_mask.clone()
    sent = done_mask.clone()
    events = {k: torch.full((episodes,), float('nan'), device=device) for k in
              ('request_s','takeoff_s','landing_s','settled_s','complete_s','fall_s','nan_s','peak_s')}
    event_metrics = [dict() for _ in range(episodes)]
    max_tilt = torch.zeros(episodes, device=device)
    max_drift = max_tilt.clone()
    traces = []
    writer = None
    if video_path:
        import imageio.v2 as imageio
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=50)
    try:
        torch.manual_seed(seed)
        obs, _ = vec.reset()
        start_position = env.scene['robot'].data.root_link_pos_w.clone()
        start_quat = env.scene['robot'].data.root_link_quat_w.clone()
        from mjlab_microduck.jump_control import yaw_from_quat
        start_yaw = yaw_from_quat(start_quat)
        judge.current_yaw[:] = start_yaw
        for step in range(round(cfg.episode_length_s/env.step_dt)):
            t = step * env.step_dt
            pre = raw_sample(env)
            pre_yaw = yaw_from_quat(pre['quat'])
            if not standing:
                pre_xy = env.scene['robot'].data.root_link_pos_w[:, :2]
                accepted = judge.press((t >= request_at) & (t <= 2.+1e-6) & ~sent & ~done_mask, pre_yaw, pre_xy)
                events['request_s'][accepted] = t
                pre_head, pre_rel = _head_metrics(env)
                for idx in accepted.nonzero().flatten().tolist():
                    event_metrics[idx]['request'] = {'heading_error_rad': 0.0, 'yaw_rate_rad_s': float(judge.yaw_rate[idx]),
                        'head_joint_error_rad': pre_head[idx].detach().cpu().tolist(), 'head_relative_rpy_rad': pre_rel[idx].detach().cpu().tolist(),
                        'foot_clearance_m': [float(x) for x in physical_trace(env)['foot_clearance'][idx]],
                        'com_z_m': float(physical_trace(env)['com_z'][idx])}
                sent |= accepted
                judge.readiness_failure |= (t >= 2.-1e-6) & ~sent & ~done_mask
                sent |= judge.readiness_failure
            obs['actor'][:,48:51] = judge.command(protocol)
            with torch.inference_mode():
                actions = policy(obs)
            obs, rewards, dones, extras = vec.step(actions)
            s = raw_sample(env)
            force = env.scene.sensors['feet_ground_contact'].data.force.reshape(episodes,-1,3).norm(dim=-1).sum(-1)
            finite = (torch.isfinite(obs['actor']).all(-1) & torch.isfinite(actions).all(-1)
                      & torch.isfinite(rewards) & torch.isfinite(env.scene['robot'].data.joint_pos).all(-1))
            # Invalidate BEFORE updating the peak, including observation/action NaNs.
            force = torch.where(finite, force, torch.full_like(force, float('nan')))
            old_peak = judge.peak.clone()
            judge.update(step+1, **s, force=force)
            head_now, rel_now = _head_metrics(env)
            physical = physical_trace(env)
            event_masks = {'takeoff': judge.took_off & torch.isnan(events['takeoff_s']) & ~done_mask,
                           'landing': judge.landed & torch.isnan(events['landing_s']) & ~done_mask,
                           'complete': judge.complete & torch.isnan(events['complete_s']) & ~done_mask,
                           'peak': (judge.peak > old_peak) & ~done_mask}
            for ename, mask in event_masks.items():
                for idx in mask.nonzero().flatten().tolist():
                    event_metrics[idx][ename] = {'heading_error_rad': float(judge.heading_error[idx]),
                        'yaw_rate_rad_s': float(judge.yaw_rate[idx]),
                        'head_joint_error_rad': head_now[idx].detach().cpu().tolist(),
                        'head_relative_rpy_rad': rel_now[idx].detach().cpu().tolist(),
                        'foot_clearance_m': physical['foot_clearance'][idx].detach().cpu().tolist(),
                        'com_z_m': float(physical['com_z'][idx])}
            failed |= ~finite | env.termination_manager.terminated | judge.invalid
            tilt = torch.rad2deg(torch.acos((1-2*s['quat'][:,1:3].square().sum(-1)).clamp(-1,1)))
            max_tilt = torch.maximum(max_tilt, torch.nan_to_num(tilt, nan=180.))
            ph = physical_trace(env)
            max_drift = torch.maximum(max_drift, (env.scene['robot'].data.root_link_pos_w-start_position)[:,:2].norm(dim=-1))
            timestamp = (step+1)*env.step_dt
            for name, flag in (('takeoff_s',judge.took_off),('landing_s',judge.landed),
                ('settled_s',judge.success),('complete_s',judge.complete),('fall_s',judge.fallen),('nan_s',judge.nan)):
                hit = flag & torch.isnan(events[name]) & ~done_mask
                events[name][hit] = timestamp
            events['peak_s'][(judge.peak>old_peak)&~done_mask] = timestamp
            if trace_path:
                traces.append({k:v.detach().cpu().numpy().copy() for k,v in
                    {**s, **ph, 'time':torch.full_like(s['z'],timestamp),
                     'active':~done_mask, 'actions':actions, 'request':judge.request,
                     'peak':judge.peak, 'joint_pos':env.scene['robot'].data.joint_pos,
                     'force':force, 'impact':judge.impact}.items()})
            if writer:
                writer.append_data(env.render())
            finish = dones.bool() | torch.full_like(done_mask, step+1 == round(cfg.episode_length_s/env.step_dt))
            robot = env.scene["robot"]
            final_delta_xy = robot.data.root_link_pos_w[:, :2] - start_position[:, :2]
            final_horizontal_speed = robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
            from mjlab_microduck.tasks import mdp as microduck_mdp
            final_pose_l1 = (
                microduck_mdp._servo_joint_pos(env, robot)
                - microduck_mdp._servo_default_joint_pos(env, robot)
            ).abs().mean(dim=-1)
            for idx in (finish & ~done_mask).nonzero().flatten().tolist():
                height_met = bool(judge.took_off[idx] and judge.peak[idx]>=judge.stand_z+judge.acceptance_delta-1e-6)
                standing_ok = bool(not failed[idx] and max_tilt[idx]<=15)
                ledger.add(idx, {'success':standing_ok if standing else bool(judge.complete[idx] and not failed[idx] and not judge.timeout[idx] and not judge.readiness_failure[idx]),
                    'standing_success':standing_ok, 'height_met':height_met,
                    'accepted':bool(judge.accepted[idx]), 'takeoff':bool(judge.took_off[idx]),
                    'landed':bool(judge.landed[idx]), 'settled':bool(judge.success[idx]),
                    'complete':bool(judge.complete[idx]), 'peak':float(judge.peak[idx]),
                    'impact':float(judge.episode_impact[idx]), 'failed':bool(failed[idx]),
                    'fallen':bool(judge.fallen[idx]), 'nan':bool(judge.nan[idx]),
                    'readiness_failure':bool(judge.readiness_failure[idx] or (not standing and not judge.accepted[idx])),
                    'request_timeout':bool(judge.timeout[idx]),
                    'max_tilt_deg':float(max_tilt[idx]), 'max_drift_m':float(max_drift[idx]),
                    'final_dx_m':float(final_delta_xy[idx, 0]),
                    'final_dy_m':float(final_delta_xy[idx, 1]),
                    'final_drift_m':float(final_delta_xy[idx].norm()),
                    'final_tilt_deg':float(tilt[idx]),
                    'final_pose_l1_rad':float(final_pose_l1[idx]),
                    'final_horizontal_speed_m_s':float(final_horizontal_speed[idx]),
                    'heading_change_rad':_wrap_scalar(float(yaw_from_quat(s['quat'][idx:idx+1])[0])-float(start_yaw[idx])),
                    'max_heading_error_rad':float(judge.max_heading_error[idx]),
                    'episode_seconds':timestamp,
                    'event_metrics': event_metrics[idx],
                    **{k:float(v[idx]) if torch.isfinite(v[idx]) else None for k,v in events.items()}})
            done_mask |= finish
            if ledger.complete:
                break
            ids = dones.nonzero().flatten()
            if len(ids):
                # Reset required by mjlab, but these slots are permanently out
                # of the sample. Their subsequent episodes can never be counted.
                fresh, _ = vec.reset() if len(ids)==episodes else env.reset(env_ids=ids)
                from tensordict import TensorDict
                obs = TensorDict(fresh, [episodes])
    finally:
        env.close()
        if writer:
            writer.close()
    if not ledger.complete:
        raise RuntimeError('Incomplete fixed evaluation sample')
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_path, **{k:np.stack([r[k] for r in traces]) for k in traces[0]})
    after_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if before_hash != after_hash:
        raise RuntimeError('Evaluated checkpoint changed')
    records = ledger.records
    peaks = [r['peak']-judge.stand_z for r in records if r['takeoff']]
    result = {'checkpoint':str(checkpoint), 'checkpoint_sha256':before_hash,
        'episodes':episodes, 'mode':mode, 'standing':standing, 'seed':seed,
        'evaluated_stage':stage, 'stand_z':judge.stand_z, 'target_delta':judge.target_delta,
        'required_delta':judge.required_delta, 'acceptance_delta':judge.acceptance_delta,
        'code_snapshot':state['run'].get('snapshot'),
        'actual_randomization':repr(asdict(cfg)),
        'peak_delta_mean':float(np.mean(peaks)) if peaks else 0.,
        'peak_delta_p10':float(np.quantile(peaks,.1)) if peaks else 0.,
        'peak_delta_p90':float(np.quantile(peaks,.9)) if peaks else 0.,
        'landing_impact_mean':float(np.mean([r['impact'] for r in records])), 'records':records}
    def finite_values(event, key):
        vals = []
        for r in records:
            value = r.get('event_metrics', {}).get(event, {}).get(key)
            if value is not None and not isinstance(value, list) and math.isfinite(float(value)):
                vals.append(float(value))
        return vals
    heading_max = [r['max_heading_error_rad'] for r in records]
    result.update({
        'heading_error_p90_deg': float(np.quantile(np.rad2deg(heading_max), .9)),
        'max_heading_error_p90_deg': float(np.quantile(np.rad2deg(heading_max), .9)),
        'max_heading_error_max_deg': float(np.max(np.rad2deg(heading_max))) if heading_max else 0.0,
        'heading_change_p90_deg': float(np.quantile(np.abs(np.rad2deg([r['heading_change_rad'] for r in records])), .9)),
        'heading_change_max_deg': float(np.max(np.abs(np.rad2deg([r['heading_change_rad'] for r in records])))) if records else 0.0,
        'yaw_rate_p90_rad_s': float(np.quantile(np.abs(finite_values('takeoff', 'yaw_rate_rad_s') or [0.0]), .9)),
        'head_dynamic_p90_deg': float(np.quantile(np.rad2deg([abs(x) for r in records for x in r.get('event_metrics', {}).get('peak', {}).get('head_joint_error_rad', [])] or [0.0]), .9)),
        'head_complete_p90_deg': float(np.quantile(np.rad2deg([abs(x) for r in records for x in r.get('event_metrics', {}).get('complete', {}).get('head_joint_error_rad', [])] or [0.0]), .9)),
        'drift_p90_m': float(np.quantile([r['max_drift_m'] for r in records], .9)),
        'final_drift_p90_m': float(np.quantile([r['final_drift_m'] for r in records], .9)),
        'final_dx_mean_m': float(np.mean([r['final_dx_m'] for r in records])),
        'final_dy_mean_m': float(np.mean([r['final_dy_m'] for r in records])),
        'final_tilt_p90_deg': float(np.quantile([r['final_tilt_deg'] for r in records], .9)),
        'final_pose_l1_p90_rad': float(np.quantile([r['final_pose_l1_rad'] for r in records], .9)),
        'final_horizontal_speed_p90_m_s': float(np.quantile(
            [r['final_horizontal_speed_m_s'] for r in records], .9)),
        'protocol': protocol,
        'nominalized': bool(mode == 'nominal'),
        'evaluator_contract': (4 if checkpoint_format == 'microduck-jump-v6'
                               else 3 if checkpoint_format == 'microduck-jump-v5' else 2),
    })
    for name,key in (('success_rate','success'),('final_success_rate','success'),('takeoff_rate','takeoff'),
            ('height_rate','height_met'),('request_acceptance_rate','accepted'),('failure_rate','failed'),
            ('standing_rate','standing_success'),('readiness_failure_rate','readiness_failure'),('timeout_rate','request_timeout')):
        result[name] = sum(r[key] for r in records)/episodes
    return result


def evaluate_v7_checkpoint(checkpoint, episodes=100, seed=123, device='cuda:0',
                           trace_path=None, video_path=None, sequence='single',
                           standing=False):
    """Evaluate V7 to episode end with one or three fixed-time requests."""
    import numpy as np
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg
    from mjlab_microduck.jump_control import JumpController, yaw_from_quat
    from mjlab_microduck.jump_curriculum import configure_v7, configure_v8
    from mjlab_microduck.jump_runner import JumpRunner
    from mjlab_microduck.tasks import mdp as microduck_mdp
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg

    if sequence not in ('single', 'triple'):
        raise ValueError("V7 sequence must be 'single' or 'triple'")
    if episodes <= 0:
        raise ValueError('episodes must be positive')

    checkpoint = Path(checkpoint)
    before_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, weights_only=False, map_location='cpu')
    if saved.get('format') not in ('microduck-jump-v7', 'microduck-jump-v8'):
        raise ValueError('V7/V8 evaluator requires a versioned jump checkpoint')
    is_v8 = saved.get('format') == 'microduck-jump-v8'
    state = saved['infos']['jump_transfer']
    base = make_microduck_jump_env_cfg(nominal_bootstrap=False, command_protocol=2)
    cfg = configure_v8(base) if is_v8 else configure_v7(base)
    cfg.seed = seed
    cfg.scene.num_envs = episodes
    cfg.auto_reset = False
    cfg.episode_length_s = 10.0 if standing else 15.0 if sequence == 'single' else 30.0
    # The evaluator owns exact request timing; the environment command stays idle.
    cfg.commands['twist'].standing_probability = 1.0
    if is_v8:
        cfg.commands['twist'].recovery_probability = 0.0

    env = ManagerBasedRlEnv(
        cfg, device=device, render_mode='rgb_array' if video_path else None
    )
    vec = RslRlVecEnvWrapper(env)
    runner = JumpRunner(
        vec, asdict(load_rl_cfg('Mjlab-Jump-Flat-MicroDuck')), device=device
    )
    runner.load(str(checkpoint), map_location=device)
    policy = runner.get_inference_policy(device=device)
    judge = JumpController(
        episodes, device, env.step_dt, state['stand_z'], state['target_delta'],
        state.get('required_delta'), policy_version=(8 if is_v8 else 7),
    )

    request_times = () if standing else ((7.0,) if sequence == 'single' else (7.0, 14.0, 21.0))
    request_total = len(request_times)
    done_mask = torch.zeros(episodes, dtype=torch.bool, device=device)
    failed = torch.zeros_like(done_mask)
    body_seen = torch.zeros_like(done_mask)
    attempted = torch.zeros(episodes, dtype=torch.long, device=device)
    accepted_count = torch.zeros_like(attempted)
    completed_count = torch.zeros_like(attempted)
    timeout_count = torch.zeros_like(attempted)
    current_request = torch.full_like(attempted, -1)
    readiness_rejections = torch.zeros_like(attempted)
    jump_peaks = torch.zeros(episodes, max(request_total, 1), device=device)
    max_tilt = torch.zeros(episodes, device=device)
    max_recovery_tilt = torch.zeros_like(max_tilt)
    recovery_watch = torch.zeros_like(done_mask)
    max_heading_error = torch.zeros_like(max_tilt)
    recovery_pose_samples = []
    traces = []
    writer = None
    if video_path:
        import imageio.v2 as imageio
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=50)

    try:
        torch.manual_seed(seed)
        obs, _ = vec.reset()
        robot = env.scene['robot']
        start_position = robot.data.root_link_pos_w.clone()
        start_yaw = yaw_from_quat(robot.data.root_link_quat_w)
        judge.current_yaw[:] = start_yaw
        steps = round(cfg.episode_length_s / env.step_dt)
        request_cursor = 0

        for step in range(steps):
            t = step * env.step_dt
            pre = raw_sample(env)
            if request_cursor < request_total and t >= request_times[request_cursor] - 1e-9:
                due = ~done_mask
                accepted = judge.press(
                    due, yaw_from_quat(pre['quat']), robot.data.root_link_pos_w[:, :2]
                )
                rejected = due & ~accepted
                attempted[due] += 1
                accepted_count[accepted] += 1
                readiness_rejections[rejected] += 1
                current_request[accepted] = request_cursor
                recovery_watch[accepted] = False
                request_cursor += 1

            obs['actor'][:, 48:51] = judge.command(2)
            with torch.inference_mode():
                actions = policy(obs)
            obs, rewards, dones, extras = vec.step(actions)
            sample_now = raw_sample(env)
            force = env.scene.sensors['feet_ground_contact'].data.force
            force = force.reshape(episodes, -1, 3).norm(dim=-1).sum(-1)
            finite = (
                torch.isfinite(obs['actor']).all(-1)
                & torch.isfinite(actions).all(-1)
                & torch.isfinite(rewards)
                & torch.isfinite(robot.data.joint_pos).all(-1)
                & torch.isfinite(robot.data.joint_vel).all(-1)
            )
            force = torch.where(finite, force, torch.full_like(force, float('nan')))
            joint_l1 = (
                microduck_mdp._servo_joint_pos(env, robot)
                - microduck_mdp._servo_default_joint_pos(env, robot)
            ).abs().mean(dim=-1)
            horizontal_speed = robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
            judge.update(
                step + 1, **sample_now, force=force, joint_l1=joint_l1,
                horizontal_speed=horizontal_speed,
            )

            valid_request = current_request >= 0
            rows = torch.arange(episodes, device=device)[valid_request]
            cols = current_request[valid_request]
            if len(rows):
                jump_peaks[rows, cols] = torch.maximum(
                    jump_peaks[rows, cols], judge.peak[valid_request]
                )

            completed = judge.complete_event.bool() & ~done_mask
            completed_count[completed] += 1
            if completed.any():
                recovery_pose_samples.extend(
                    float(x) for x in joint_l1[completed].detach().cpu()
                )
            timed_out = judge.timeout_event.bool() & ~done_mask
            timeout_count[timed_out] += 1

            tilt = torch.rad2deg(torch.acos(
                (1.0 - 2.0 * sample_now['quat'][:, 1:3].square().sum(-1)).clamp(-1, 1)
            ))
            max_tilt = torch.maximum(max_tilt, torch.nan_to_num(tilt, nan=180.0))
            # Continue watching after the completion event until the next
            # accepted request or episode end. A policy cannot hide a delayed
            # wobble merely by collecting its five-second completion payment.
            recovery_watch |= judge.impact_paid & (current_request >= 0)
            recovery = recovery_watch & ~done_mask
            max_recovery_tilt[recovery] = torch.maximum(
                max_recovery_tilt[recovery], torch.nan_to_num(tilt[recovery], nan=180.0)
            )
            max_heading_error = torch.maximum(
                max_heading_error, judge.max_heading_error
            )
            body_seen |= sample_now['body']
            failed |= (
                ~finite | env.termination_manager.terminated | judge.invalid
            )

            timestamp = (step + 1) * env.step_dt
            if trace_path:
                physical = physical_trace(env)
                traces.append({
                    k: v.detach().cpu().numpy().copy()
                    for k, v in {
                        **sample_now, **physical,
                        'time': torch.full_like(sample_now['z'], timestamp),
                        'active': ~done_mask,
                        'actions': actions,
                        'request': judge.request,
                        'busy': judge.busy,
                        'complete_event': judge.complete_event,
                        'peak': judge.peak,
                        'joint_pos': robot.data.joint_pos,
                        'force': force,
                        'impact': judge.impact,
                    }.items()
                })
            if writer:
                writer.append_data(env.render())

            finish = dones.bool() | torch.full_like(done_mask, step + 1 == steps)
            done_mask |= finish
            if done_mask.all():
                break
            ids = dones.nonzero().flatten()
            if len(ids):
                # Terminated slots are reset only to satisfy the vector wrapper;
                # done_mask permanently excludes them from later requests/results.
                fresh, _ = vec.reset() if len(ids) == episodes else env.reset(env_ids=ids)
                from tensordict import TensorDict
                obs = TensorDict(fresh, [episodes])

        final_position = robot.data.root_link_pos_w
        final_xy = final_position[:, :2] - start_position[:, :2]
        final_speed = robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
        final_quat = robot.data.root_link_quat_w
        final_tilt = torch.rad2deg(torch.acos(
            (1.0 - 2.0 * final_quat[:, 1:3].square().sum(-1)).clamp(-1, 1)
        ))
        final_pose = (
            microduck_mdp._servo_joint_pos(env, robot)
            - microduck_mdp._servo_default_joint_pos(env, robot)
        ).abs().mean(dim=-1)
        heading_change = torch.rad2deg(torch.abs(torch.atan2(
            torch.sin(yaw_from_quat(final_quat) - start_yaw),
            torch.cos(yaw_from_quat(final_quat) - start_yaw),
        )))

        required = request_total
        sequence_success = (
            (completed_count == required)
            & (accepted_count == required)
            & (timeout_count == 0)
            & (readiness_rejections == 0)
            & ~failed
        ) if not standing else (~failed & ~body_seen & (max_tilt <= 3.0))
        records = []
        for idx in range(episodes):
            records.append({
                'episode_id': idx,
                'sequence_success': bool(sequence_success[idx]),
                'requested_jumps': required,
                'accepted_jumps': int(accepted_count[idx]),
                'completed_jumps': int(completed_count[idx]),
                'timeout_count': int(timeout_count[idx]),
                'readiness_rejections': int(readiness_rejections[idx]),
                'failed': bool(failed[idx]),
                'fallen': bool(judge.fallen[idx]),
                'body_contact': bool(body_seen[idx]),
                'invalid': bool(judge.invalid[idx]),
                'max_tilt_deg': float(max_tilt[idx]),
                'recovery_tilt_max_deg': float(max_recovery_tilt[idx]),
                'final_drift_m': float(final_xy[idx].norm()),
                'final_heading_change_deg': float(heading_change[idx]),
                'final_tilt_deg': float(final_tilt[idx]),
                'final_pose_l1_rad': float(final_pose[idx]),
                'final_horizontal_speed_m_s': float(final_speed[idx]),
                'peak_deltas_m': [
                    max(0.0, float(jump_peaks[idx, j] - judge.stand_z))
                    for j in range(required)
                ],
                'episode_seconds': cfg.episode_length_s,
            })
    finally:
        env.close()
        if writer:
            writer.close()

    if trace_path and traces:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trace_path,
            **{key: np.stack([row[key] for row in traces]) for key in traces[0]},
        )
    after_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if before_hash != after_hash:
        raise RuntimeError('Evaluated checkpoint changed')

    requested_denominator = max(episodes * request_total, 1)
    peak_deltas = [
        delta for record in records for delta in record['peak_deltas_m']
    ] or [0.0]
    recovery_pose_values = recovery_pose_samples or [1.0]
    result = {
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': before_hash,
        'episodes': episodes,
        'seconds': cfg.episode_length_s,
        'mode': 'nominal',
        'sequence': sequence,
        'standing': standing,
        'seed': seed,
        'protocol': 2,
        'policy_version': 8 if is_v8 else 7,
        'evaluator_contract': 8 if is_v8 else 7,
        'stand_z': judge.stand_z,
        'target_delta': judge.target_delta,
        'required_delta': judge.required_delta,
        'acceptance_delta': judge.acceptance_delta,
        'success_rate': float(sequence_success.float().mean()),
        'sequence_success_rate': float(sequence_success.float().mean()),
        'jump_success_rate': float(completed_count.sum()) / requested_denominator,
        'request_acceptance_rate': float(accepted_count.sum()) / requested_denominator,
        'failure_rate': float(failed.float().mean()),
        'fallen_rate': float(judge.fallen.float().mean()),
        'body_contact_rate': float(body_seen.float().mean()),
        'invalid_rate': float(judge.invalid.float().mean()),
        'timeout_rate': float((timeout_count > 0).float().mean()),
        'readiness_failure_rate': float((readiness_rejections > 0).float().mean()),
        'peak_delta_p10': float(np.quantile(peak_deltas, .1)),
        'peak_delta_p90': float(np.quantile(peak_deltas, .9)),
        'final_drift_p90_m': float(np.quantile(
            [row['final_drift_m'] for row in records], .9
        )),
        'final_drift_max_m': float(max(row['final_drift_m'] for row in records)),
        'heading_change_p90_deg': float(np.quantile(
            [row['final_heading_change_deg'] for row in records], .9
        )),
        'heading_change_max_deg': float(max(
            row['final_heading_change_deg'] for row in records
        )),
        'max_heading_error_p90_deg': float(np.quantile(
            torch.rad2deg(max_heading_error).detach().cpu().numpy(), .9
        )),
        'recovery_tilt_p90_deg': float(np.quantile(
            [row['recovery_tilt_max_deg'] for row in records], .9
        )),
        'recovery_tilt_max_deg': float(max(
            row['recovery_tilt_max_deg'] for row in records
        )),
        'recovery_pose_l1_p90_rad': float(np.quantile(recovery_pose_values, .9)),
        'final_pose_l1_p90_rad': float(np.quantile(
            [row['final_pose_l1_rad'] for row in records], .9
        )),
        'final_horizontal_speed_p90_m_s': float(np.quantile(
            [row['final_horizontal_speed_m_s'] for row in records], .9
        )),
        'final_horizontal_speed_max_m_s': float(max(
            row['final_horizontal_speed_m_s'] for row in records
        )),
        'records': records,
    }
    return result
