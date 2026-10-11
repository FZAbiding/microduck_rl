"""Strict V9 evaluation of both legacy source actors and V9 candidates."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import math

import numpy as np
import torch

from mjlab_microduck.jump_artifacts import sha256
from mjlab_microduck.jump_control import JumpController, yaw_from_quat
from mjlab_microduck.jump_v9 import configure_v9, validate_metrics


def height_failure_breakdown(records):
    """Separate the height gate from completion; timeouts alone cannot do this."""
    requests = sum(r['requested_jumps'] for r in records)
    qualified = sum(p >= .030 - 1e-6 for r in records for p in r['peak_deltas_m'])
    completed = sum(r['completed_jumps'] for r in records)
    return dict(requested_jumps=requests, height_qualified_requests=qualified,
        height_shortfall_requests=requests-qualified,
        height_qualified_uncompleted_requests=qualified-completed,
        requested_height_shortfall_rate=(requests-qualified)/requests if requests else None)


def summarize(records, sequence):
    """Terminal snapshots only. Missing event measurements remain unavailable."""
    def quantile(key, q):
        values = [r[key] for r in records if r[key] is not None]
        return float(np.quantile(values, q)) if values else None

    def rate(key):
        return sum(bool(r[key]) for r in records) / len(records)

    requests = sum(r['requested_jumps'] for r in records)
    completed = sum(r['completed_jumps'] for r in records)
    peaks = [p for r in records for p in r['peak_deltas_m']]
    result = dict(evaluator_contract=9, sequence=sequence, episodes=len(records),
        success_rate=rate('success'), sequence_success_rate=rate('success'),
        jump_success_rate=completed / requests if requests else 0.,
        request_acceptance_rate=sum(r['accepted_jumps'] for r in records) / requests if requests else 1.,
        failure_rate=rate('failed'), body_contact_rate=rate('body_contact'),
        invalid_rate=rate('invalid'), timeout_rate=rate('timeout'),
        recovery_timeout_rate=rate('recovery_timeout'),
        readiness_failure_rate=rate('rejected'), busy_stuck_rate=rate('busy_stuck'),
        peak_delta_p10=float(np.quantile(peaks, .1)) if peaks else None,
        records=records)
    for source, p90, maximum in (
        ('final_drift_m', 'final_drift_p90_m', 'final_drift_max_m'),
        ('heading_change_deg', 'heading_change_p90_deg', 'heading_change_max_deg'),
        ('recovery_tilt_deg', 'recovery_tilt_p90_deg', 'recovery_tilt_max_deg'),
        ('foot_pose_deg', 'foot_pose_p90_deg', 'foot_pose_max_deg'),
        ('ankle_error_rad', 'ankle_error_p90_rad', 'ankle_error_max_rad'),
        ('leg_home_max_rad', 'leg_home_max_p90_rad', 'leg_home_max_rad'),
        ('ready_latency_s', 'ready_latency_p90_s', 'ready_latency_max_s'),
        ('complete_latency_s', 'complete_latency_p90_s', 'complete_latency_max_s'),
    ):
        result[p90], result[maximum] = quantile(source, .9), quantile(source, 1.)
    result['final_horizontal_speed_max_m_s'] = quantile('final_horizontal_speed', 1.)
    result['max_heading_error_p90_deg'] = quantile('max_heading_error_deg', .9)
    result['recovery_observed_episodes'] = sum(r['recovery_tilt_deg'] is not None for r in records)
    result['height_failure_breakdown'] = height_failure_breakdown(records)
    validate_metrics(result)
    return result


def evaluate(checkpoint, episodes=100, seed=123, device='cuda:0', sequence='single',
             trace_path=None, video_path=None, long_interval=False,
             landing_bank_out=None, recovery_bank=None):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg
    from mjlab_microduck.jump_runner import JumpRunner
    from mjlab_microduck.jump_evaluation import raw_sample
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg

    if episodes < 1 or sequence not in ('single', 'triple', 'standing', 'recovery'):
        raise ValueError('invalid evaluation request')
    if landing_bank_out and sequence != 'single':
        raise ValueError('landing collection requires single sequence')
    if recovery_bank and sequence != 'recovery':
        raise ValueError('landing bank evaluation requires recovery sequence')
    checkpoint = Path(checkpoint)
    digest = sha256(checkpoint)
    cfg = configure_v9(make_microduck_jump_env_cfg(nominal_bootstrap=False, command_protocol=2))
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = episodes, seed, False
    # 7 s request + launch/flight + up to 8 s recovery must finish before cutoff.
    seconds = 40. if sequence == 'triple' else 20. if sequence == 'single' else 15.
    if long_interval and sequence == 'triple':
        seconds = 42.
    cfg.episode_length_s = seconds
    if recovery_bank:
        cfg.events['jump_reset_state'].params['recovery_bank_path'] = str(recovery_bank)
    command = cfg.commands['twist']
    command.standing_probability = 0. if sequence == 'recovery' else 1.
    command.recovery_probability = 1. if sequence == 'recovery' else 0.
    command.single_probability = command.triple_probability = 0.
    env = ManagerBasedRlEnv(cfg, device=device, render_mode='rgb_array' if video_path else None)
    writer = None
    traces = []
    landing_rows = []
    landing_offsets = (.5, 2., 5.)
    landing_saved = torch.zeros(episodes, len(landing_offsets), dtype=torch.bool, device=device)
    try:
        vec = RslRlVecEnvWrapper(env)
        agent = load_rl_cfg('Mjlab-Jump-Flat-MicroDuck')
        agent.init_checkpoint = None
        runner = JumpRunner(vec, asdict(agent), device=device)
        runner.load(str(checkpoint), map_location=device)
        internal = mdp.jump_controller(env)
        internal.policy_version = 9
        internal.target_delta, internal.required_delta = .032, .030
        internal.strict_first_request = True
        policy = runner.get_inference_policy(device=device)
        torch.manual_seed(seed)
        env.seed(seed)
        obs, _ = vec.reset()
        judge = JumpController(episodes, device, env.step_dt, internal.stand_z, .032, .030, 9)
        robot = env.scene['robot']
        start_xy = robot.data.root_link_pos_w[:, :2].clone()
        start_yaw = yaw_from_quat(robot.data.root_link_quat_w).clone()
        if sequence == 'recovery':
            judge.start_recovery(torch.arange(episodes, device=device), start_yaw, start_xy)
            if recovery_bank:
                judge.target_yaw.copy_(internal.target_yaw)
                judge.heading_error.copy_(internal.heading_error)
        times = ((7.,) if sequence == 'single' else
                 ((7., 17., 27.) if long_interval else (7., 14., 21.)) if sequence == 'triple' else ())
        count = len(times)
        active = torch.ones(episodes, dtype=torch.bool, device=device)
        failed = ~active.clone()
        body_seen = failed.clone()
        rejected = failed.clone()
        timeout_seen = failed.clone()
        recovery_timeout_seen = failed.clone()
        accepted = torch.zeros(episodes, dtype=torch.long, device=device)
        completed = accepted.clone()
        request_index = torch.full_like(accepted, -1)
        peak = torch.zeros(episodes, max(count, 1), device=device)
        contact_at = torch.full_like(peak, float('nan'))
        ready_latency = contact_at.clone()
        complete_latency = contact_at.clone()
        recovery_max_tilt = torch.full((episodes,), float('nan'), device=device)
        max_heading = torch.zeros(episodes, device=device)
        final = {}
        records = [None] * episodes
        cursor = 0
        if sequence == 'recovery':
            contact_at[:, 0] = 0.
            request_index[:] = 0
        if video_path:
            import imageio.v2 as imageio
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video_path, fps=50)

        for step in range(round(seconds / env.step_dt)):
            t = step * env.step_dt
            if cursor < count and t >= times[cursor] - 1e-9:
                got = judge.press(active, yaw_from_quat(robot.data.root_link_quat_w),
                                  robot.data.root_link_pos_w[:, :2])
                rejected |= active & ~got
                judge.mark_readiness_failure(active & ~got)
                accepted += got.long()
                request_index[got] = cursor
                cursor += 1
            obs['actor'][:, 48:51] = judge.command(2)
            with torch.inference_mode():
                action = policy(obs)
            obs, reward, dones, _ = vec.step(action)
            sample = raw_sample(env)
            metrics = mdp.jump_v9_measurements(env)
            finite = (torch.isfinite(action).all(-1) & torch.isfinite(obs['actor']).all(-1)
                      & torch.isfinite(reward) & torch.isfinite(robot.data.joint_vel).all(-1))
            force = env.scene.sensors['feet_ground_contact'].data.force.reshape(episodes, -1, 3).norm(dim=-1).sum(-1)
            force = torch.where(finite, force, torch.full_like(force, float('nan')))
            judge.update(step+1, **sample, force=force, **metrics)
            stamp = (step+1) * env.step_dt
            failed |= active & (~finite | judge.invalid | env.termination_manager.terminated)
            body_seen |= active & sample['body']
            timeout_seen |= active & judge.timeout
            recovery_timeout_seen |= active & judge.recovery_timeout
            completed += (active & judge.complete_event.bool()
                          & ~judge.recovery_timeout & ~judge.request_timeout).long()
            valid = active & (request_index >= 0)
            rows = valid.nonzero().flatten()
            cols = request_index[rows]
            if len(rows):
                peak[rows, cols] = torch.maximum(peak[rows, cols], judge.peak[rows])
                touch = judge.impact_paid[rows] & torch.isnan(contact_at[rows, cols])
                contact_at[rows[touch], cols[touch]] = stamp
                elapsed = stamp - contact_at[rows, cols]
                ready = (judge.ready_time[rows] >= 2.-1e-4) & torch.isnan(ready_latency[rows, cols]) & torch.isfinite(elapsed)
                ready_latency[rows[ready], cols[ready]] = elapsed[ready]
                done_event = judge.complete_event[rows].bool() | judge.recovery_complete_event[rows].bool()
                complete_latency[rows[done_event], cols[done_event]] = elapsed[done_event]
            if landing_bank_out:
                elapsed = stamp - contact_at[:, 0]
                safe_landing = (active & ~failed & ~body_seen & sample['feet'].all(-1)
                    & judge.landed & (judge.tilt <= math.radians(6))
                    & (metrics['joint_max'] <= .15) & (metrics['horizontal_speed'] <= .03)
                    & (sample['omega_xy'].norm(dim=-1) <= .5)
                    & (metrics['foot_pitch_max'] <= math.radians(6)))
                for j, offset in enumerate(landing_offsets):
                    eligible = safe_landing & (elapsed >= offset) & ~landing_saved[:, j]
                    for i in eligible.nonzero().flatten().tolist():
                        qpos = env.sim.data.qpos[i].detach().cpu().numpy().copy()
                        qpos[:3] -= env.scene.terrain.env_origins[i].cpu().numpy()
                        landing_rows.append(dict(qpos=qpos,
                            qvel=env.sim.data.qvel[i].detach().cpu().numpy().copy(),
                            joint_max=np.asarray(float(metrics['joint_max'][i])),
                            episode_id=np.asarray(i), offset_s=np.asarray(float(elapsed[i])),
                            action=env.action_manager.action[i].detach().cpu().numpy().copy(),
                            prev_action=env.action_manager.prev_action[i].detach().cpu().numpy().copy(),
                            prev_prev_action=env.action_manager.prev_prev_action[i].detach().cpu().numpy().copy(),
                            target_yaw=np.asarray(float(judge.target_yaw[i])),
                            heading_error=np.asarray(float(judge.heading_error[i]))))
                    landing_saved[eligible, j] = True
            recovery = valid & judge.impact_paid
            tilt = torch.rad2deg(judge.tilt)
            recovery_max_tilt[recovery] = torch.maximum(
                torch.nan_to_num(recovery_max_tilt[recovery]), tilt[recovery])
            max_heading[active] = torch.maximum(max_heading[active], torch.rad2deg(judge.max_heading_error[active]))
            foot = torch.rad2deg(torch.maximum(metrics['foot_pitch_max'], metrics['foot_roll_max']))
            # Freeze final physical values at the first terminal step of each slot.
            heading = torch.rad2deg(torch.atan2(
                torch.sin(yaw_from_quat(sample['quat'])-start_yaw),
                torch.cos(yaw_from_quat(sample['quat'])-start_yaw)).abs())
            physical = dict(final_drift_m=(robot.data.root_link_pos_w[:, :2]-start_xy).norm(dim=-1),
                heading_change_deg=heading, final_horizontal_speed=metrics['horizontal_speed'],
                foot_pose_deg=foot, ankle_error_rad=metrics['ankle_max'],
                leg_home_max_rad=metrics['joint_max'], final_tilt_deg=tilt)
            for key, value in physical.items():
                if key not in final:
                    final[key] = value.clone()
                final[key][active] = value[active]
            if trace_path:
                traces.append({k: v.detach().cpu().numpy().copy() for k, v in dict(
                    time=torch.full_like(tilt, stamp), active=active, actions=action,
                    **sample, **metrics, ready_time=judge.ready_time,
                    complete_event=judge.complete_event, recovery_complete_event=judge.recovery_complete_event,
                    **physical).items()})
            if writer:
                writer.append_data(env.render())
            finishing = active & (dones.bool() | failed | (step+1 == round(seconds/env.step_dt)))
            for idx in finishing.nonzero().flatten().tolist():
                def finite_max(values):
                    values = values[torch.isfinite(values)]
                    return float(values.max()) if len(values) else None
                success = (int(completed[idx]) == count and int(accepted[idx]) == count and
                           not bool(rejected[idx] | timeout_seen[idx] | recovery_timeout_seen[idx]))
                if sequence == 'standing':
                    success = bool(judge.ready_time[idx] >= 2.-1e-4)
                elif sequence == 'recovery':
                    success = bool(judge.recovery_complete[idx] & ~judge.recovery_timeout[idx])
                success &= not bool(failed[idx] | body_seen[idx])
                row = dict(episode_id=idx, success=success, requested_jumps=count,
                    accepted_jumps=int(accepted[idx]), completed_jumps=int(completed[idx]),
                    rejected=bool(rejected[idx]), failed=bool(failed[idx]),
                    body_contact=bool(body_seen[idx]), invalid=bool(judge.invalid[idx]),
                    timeout=bool(timeout_seen[idx]), recovery_timeout=bool(recovery_timeout_seen[idx]),
                    busy_stuck=bool(judge.busy[idx] and not failed[idx]), terminal_time_s=stamp,
                    peak_deltas_m=[max(0.,float(v)-judge.stand_z) for v in peak[idx, :count]],
                    recovery_tilt_deg=finite_max(recovery_max_tilt[idx:idx+1]),
                    ready_latency_s=finite_max(ready_latency[idx]),
                    complete_latency_s=finite_max(complete_latency[idx]),
                    max_heading_error_deg=float(max_heading[idx]))
                row.update({key: float(value[idx]) if torch.isfinite(value[idx]) else None
                            for key, value in final.items()})
                records[idx] = row
            active &= ~finishing
            if not active.any():
                break
            reset_ids = dones.nonzero().flatten()
            if len(reset_ids):
                # Completed records are immutable even when their slots must reset.
                fresh, _ = env.reset(env_ids=reset_ids)
                from tensordict import TensorDict
                obs = TensorDict(fresh, [episodes])
        result = summarize(records, sequence)
        result.update(checkpoint=str(checkpoint), checkpoint_sha256=digest, seed=seed,
                      seconds=seconds, request_times=list(times), stand_z=judge.stand_z,
                      target_delta=.032, required_delta=.030, mode='nominal')
        if recovery_bank:
            result['recovery_bank'] = dict(path=str(recovery_bank), sha256=sha256(recovery_bank),
                initialization='bank-versioned physical/action/yaw state; BAM/solver histories reset')
        if landing_bank_out:
            from mjlab_microduck.jump_landing_bank import save_bank
            result['landing_bank'] = save_bank(landing_bank_out, landing_rows, robot.joint_names,
                dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, seed=seed,
                     episodes=episodes, offsets_s=landing_offsets, stand_z=judge.stand_z,
                     initialization='physical qpos/qvel plus action history and original yaw target; BAM/solver histories reset'))
    finally:
        env.close()
        if writer:
            writer.close()
    if sha256(checkpoint) != digest:
        raise RuntimeError('checkpoint changed during evaluation')
    if trace_path and traces:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_path, **{k: np.stack([r[k] for r in traces]) for k in traces[0]})
    return result
