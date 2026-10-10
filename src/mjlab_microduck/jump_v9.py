"""V9 experiment contracts: explicit stages, finite metrics, immutable parents."""
from __future__ import annotations

import math

from mjlab_microduck.jump_curriculum import configure_v8

MAX_UPDATES = 4000
STAGES = {
    # standing, recovery, single, triple; noise; quality weight multiplier
    'pilot': ((.30, .10, .60, .00), .25, .25),
    'recovery': ((.20, .35, .45, .00), .50, .50),
    'single': ((.20, .20, .60, .00), .50, .50),
    'triple': ((.20, .15, .35, .30), .50, .75),
    'consolidate': ((.15, .10, .15, .60), 1., 1.),
}


def configure_v9(cfg, stage='pilot', strict_first_request=True):
    from mjlab.managers import EventTermCfg, RewardTermCfg
    from mjlab_microduck.tasks import mdp
    cfg = configure_v8(cfg)
    dist, noise, weight = STAGES[stage]
    command = cfg.commands['twist']
    command.policy_version = 9
    command.strict_first_request = strict_first_request
    command.recovery_noise_scale = noise
    (command.standing_probability, command.recovery_probability,
     command.single_probability, command.triple_probability) = dist
    command.request_times = (7., 14., 21.) if stage == 'consolidate' else (7., 17., 27.)
    command.request_jitter_s = .25
    command.resampling_time_range = (1000., 1000.)
    cfg.episode_length_s = 40. if stage == 'consolidate' else 42.
    cfg.events['jump_reset_state'] = EventTermCfg(func=mdp.jump_v9_reset, mode='reset')
    cfg.rewards['jump_recovery_complete'] = RewardTermCfg(
        func=mdp.jump_v2_reward, weight=4., params={'quantity': 'recovery_complete_event'})
    for name in ('jump_settle_leg_home', 'jump_settle_foot_pose',
                 'jump_settle_upright', 'jump_settle_single_foot'):
        cfg.rewards[name].weight *= weight
    # Correct event units first, then start with a deliberately small event mass.
    cfg.rewards['jump_touchdown_displacement'].weight = -.125
    return cfg


def set_learning_rate(algorithm, value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError('learning rate must be finite and positive')
    algorithm.learning_rate = value
    for group in algorithm.optimizer.param_groups:
        group['lr'] = value
    if algorithm.learning_rate != value or any(
            g['lr'] != value for g in algorithm.optimizer.param_groups):
        raise RuntimeError('optimizer did not apply learning rate')


def set_idle_exploration(actor, joint_names, scale):
    """Absolute idle noise relative to the original recipe; safe across resumes."""
    import torch
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('idle std scale must be finite and in (0, 1]')
    if len(joint_names) != 14 or len(set(joint_names)) != 14:
        raise ValueError('Expected 14 named servo actions')
    head = [any(k in n for k in ('head_', 'neck_')) for n in joint_names]
    if sum(head) != 4:
        raise ValueError('Expected four head/neck actions')
    values = actor.idle_std.new_tensor([.03 if h else .05 for h in head]) * scale
    with torch.no_grad():
        actor.idle_std.copy_(values)
    return actor.idle_std.detach().cpu().tolist()


REQUIRED_METRICS = (
    'success_rate', 'sequence_success_rate', 'jump_success_rate',
    'request_acceptance_rate', 'failure_rate', 'body_contact_rate', 'invalid_rate',
    'timeout_rate', 'recovery_timeout_rate', 'readiness_failure_rate',
    'busy_stuck_rate', 'peak_delta_p10', 'final_drift_p90_m', 'final_drift_max_m',
    'heading_change_p90_deg', 'heading_change_max_deg',
    'recovery_tilt_p90_deg', 'recovery_tilt_max_deg',
    'foot_pose_p90_deg', 'foot_pose_max_deg', 'ankle_error_p90_rad',
    'ankle_error_max_rad', 'leg_home_max_p90_rad', 'leg_home_max_rad',
    'final_horizontal_speed_max_m_s', 'ready_latency_p90_s', 'ready_latency_max_s',
    'complete_latency_p90_s', 'complete_latency_max_s',
)


def validate_metrics(metrics):
    if metrics.get('evaluator_contract') != 9:
        raise ValueError('requires V9 evaluation contract')
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise ValueError(f'missing V9 metrics: {missing}')
    for key in REQUIRED_METRICS:
        value = metrics[key]
        if key.endswith('_rate') and (value is None or not isinstance(value, (float, int))
                                    or not 0 <= value <= 1):
            raise ValueError(f'invalid V9 rate: {key}')
        # Unobserved events are unavailable, never silently ideal zero.
        if value is not None and (not isinstance(value, (float, int)) or not math.isfinite(value)):
            raise ValueError(f'invalid V9 metric: {key}')


def safe_candidate(metrics):
    validate_metrics(metrics)
    return all(metrics[key] == 0 for key in ('failure_rate', 'body_contact_rate', 'invalid_rate'))


def selection_key(single, triple):
    for metrics in (single, triple):
        if not safe_candidate(metrics):
            return None
        if metrics['final_drift_p90_m'] is None or metrics['peak_delta_p10'] is None:
            return None
    return (triple['sequence_success_rate'], single['success_rate'],
            -max(single['final_drift_p90_m'], triple['final_drift_p90_m']))


def update_best(state, checkpoint, single, triple):
    """Compare to the previous best before mutation; never promote unsafe runs."""
    key = selection_key(single, triple)
    old = state.get('best_key')
    regression = key is None or (old is not None and key[0] < old[0] - .10)
    if key is not None and (old is None or key > tuple(old)):
        state.update(best_checkpoint=str(checkpoint), best_key=list(key))
    if regression:
        state['learning_rate'] = min(state['learning_rate'], 2.5e-5)
        state['checkpoint'] = state.get('best_checkpoint')
        state['status'] = 'diagnose' if state['checkpoint'] is None else 'rollback'
    else:
        state['checkpoint'] = str(checkpoint)
        state['status'] = 'continue'
    return state['status']


def worker_arguments(branch, output, count=50, num_envs=4096, device='cuda:0'):
    """There is exactly one resume pointer, including after a rollback."""
    from mjlab_microduck.jump_artifacts import sha256
    checkpoint = branch['checkpoint']
    if checkpoint is None:
        raise ValueError('branch has no resume checkpoint')
    return ['scripts/train_jump_v9.py', '--resume', str(checkpoint),
            '--parent-sha256', sha256(checkpoint), '--output', str(output),
            '--max-iterations', str(count), '--num-envs', str(num_envs),
            '--learning-rate', str(branch['learning_rate']), '--seed', '42',
            '--device', device, '--stage', 'pilot', '--export',
            '--idle-std-scale', str(branch.get('idle_std_scale', 1.))]


def blocked_jump_experience(progress):
    """Fail a pilot data gate, not a policy-quality gate, after the request window."""
    masses = progress['reward_mass_per_env']
    return (progress['measured_steps'] >= 500
            and masses['jump_readiness_failure'] < 0
            and masses['jump_height_progress'] == 0
            and masses['jump_complete'] == 0
            and masses['jump_recovery_complete'] == 0)
