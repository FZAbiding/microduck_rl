#!/usr/bin/env python3
"""One 50-update physical-landing recovery experiment, with frozen quality gates."""
import argparse
import json
from pathlib import Path

from mjlab_microduck.jump_artifacts import atomic_json, sha256
from mjlab_microduck.jump_v9 import MAX_UPDATES, safe_candidate, selection_key, validate_metrics
from supervise_jump_v9 import evaluation, read, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('artifacts/jump-v9'))
    p.add_argument('--label', default='A-landing-0050')
    p.add_argument('--smoke-result', type=Path, required=True)
    p.add_argument('--export-check', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if Path(a.label).name != a.label or not a.label.startswith('A-landing-'):
        p.error('use a fresh A-landing-* label')
    root, state_path = a.root, a.root/'state.json'
    banks = root/'landing-recovery'
    state = read(state_path)
    branch = state['branches']['A']
    parent = branch['checkpoint']
    parent_hash = sha256(parent)
    output = root/a.label
    if state['status'] not in ('diagnose', 'pilot_checkpoint') or output.exists():
        raise RuntimeError('state is busy or attempt exists; no implicit retry')
    smoke, export = read(a.smoke_result), read(a.export_check)
    config = read(a.smoke_result.parent/'configuration.json')['arguments']
    train_bank, heldout_bank = banks/'train-history-seed42.npz', banks/'eval-history-seed124.npz'
    train_baseline, heldout_baseline = (read(banks/f'baseline-history-{name}-123.json') for name in ('train', 'heldout'))
    for baseline, bank in ((train_baseline, train_bank), (heldout_baseline, heldout_bank)):
        validate_metrics(baseline)
        if not (safe_candidate(baseline) and baseline['episodes'] >= 100
                and baseline['checkpoint_sha256'] == parent_hash
                and baseline['recovery_bank']['sha256'] == sha256(bank)):
            raise RuntimeError('unsafe/mismatched landing reset baseline')
    if not (smoke['block_completed_updates'] == 5 and config['num_envs'] == 64
            and smoke['stage'] == 'landing_recovery' and smoke['parent_sha256'] == parent_hash
            and config['learning_rate'] == 2.5e-5 and config['idle_std_scale'] == .25
            and config['recovery_bank_max_joint_error'] == .11
            and sha256(config['recovery_bank']) == sha256(train_bank)
            and not config['warm_first_request'] and export['passed']
            and export['checkpoint_sha256'] == smoke['checkpoint_sha256']
            and str(a.smoke_result) in state['smoke_results']):
        raise RuntimeError('landing recipe smoke/export gate mismatch')
    if state['reserved_updates'] + 50 > MAX_UPDATES:
        raise RuntimeError('budget exceeded')
    old = branch['best_metrics']
    attempt = dict(label=a.label, parent_checkpoint=parent, parent_sha256=parent_hash,
        learning_rate=2.5e-5, idle_std_scale=.25, stage='landing_recovery',
        train_bank_sha256=sha256(train_bank), heldout_bank_sha256=sha256(heldout_bank),
        frontier_joint_max=.11, reserved_updates=50, status='running')
    state['attempts'].append(attempt)
    state['reserved_updates'] += 50
    state['status'] = 'training'
    atomic_json(state_path, state)
    try:
        run(['scripts/train_jump_v9.py', '--resume', parent, '--parent-sha256', parent_hash,
             '--output', str(output), '--stage', 'landing_recovery', '--max-iterations', '50',
             '--num-envs', '4096', '--learning-rate', '2.5e-5', '--idle-std-scale', '.25',
             '--recovery-bank', str(train_bank), '--recovery-bank-max-joint-error', '.11',
             '--seed', '42', '--device', a.device, '--export'], output.with_suffix('.log'))
        result = read(output/'result.json')
        if not (result['parent_sha256'] == parent_hash and result['block_completed_updates'] == 50
                and result['actual_learning_rate'] == 2.5e-5
                and result['checkpoint_sha256'] == sha256(result['checkpoint'])):
            raise RuntimeError('worker provenance mismatch')
        checkpoint = result['checkpoint']
        state['completed_updates'] += 50
        state['status'] = 'evaluating'
        attempt.update(status='trained', completed_updates=50, checkpoint=checkpoint)
        atomic_json(state_path, state)
        metrics = {}
        for sequence in ('standing', 'single', 'triple', 'recovery'):
            metrics[sequence] = evaluation(root, checkpoint, sequence, a.device, a.label)
            attempt['metrics'] = {name: {k: v for k, v in value.items() if k != 'records'}
                                  for name, value in metrics.items()}
            atomic_json(state_path, state)
        heldout_out = root/'evaluations'/f'{a.label}-heldout-recovery-123.json'
        if heldout_out.exists():
            raise RuntimeError('refusing stale heldout evaluation')
        run(['scripts/evaluate_jump_v9.py', '--checkpoint', checkpoint, '--sequence', 'recovery',
             '--episodes', '100', '--seed', '123', '--recovery-bank', str(heldout_bank),
             '--json-out', str(heldout_out), '--device', a.device], heldout_out.with_suffix('.log'))
        heldout = read(heldout_out)
        validate_metrics(heldout)
        if heldout['checkpoint_sha256'] != sha256(checkpoint) or heldout['recovery_bank']['sha256'] != sha256(heldout_bank):
            raise RuntimeError('heldout checkpoint/bank provenance mismatch')
        attempt['heldout_metrics'] = {k:v for k,v in heldout.items() if k != 'records'}
        regression = (not all(safe_candidate(m) for m in (*metrics.values(), heldout))
            or metrics['standing']['success_rate'] < .99
            or any(metrics[s]['success_rate'] < old[s]['success_rate']-.10-1e-9
                   for s in ('single','triple','recovery')))
        if regression:
            confirm = {s: evaluation(root, checkpoint, s, a.device, a.label+'-confirm')
                       for s in ('single','triple')}
            attempt['confirmation'] = {s:m['success_rate'] for s,m in confirm.items()}
        heldout_gain = heldout['success_rate'] - heldout_baseline['success_rate']
        key = selection_key(metrics['single'], metrics['triple'])
        if not regression and heldout_gain >= .05-1e-9:
            branch.update(checkpoint=checkpoint, learning_rate=2.5e-5,
                          status='continue', idle_std_scale=.25)
            branch.pop('reason', None)
            if key is not None and key > tuple(branch['best_key']):
                branch.update(best_checkpoint=checkpoint, best_key=key, best_metrics=metrics)
            decision = 'continue'
        else:
            # Preserve the measured best and stop this recipe, even with budget remaining.
            branch.update(checkpoint=branch['best_checkpoint'], status='diagnose', learning_rate=2.5e-5,
                          reason='landing_curriculum_regressed' if regression else 'no_heldout_recovery_gain')
            decision = 'rollback' if regression else 'diagnose'
        attempt.update(status='evaluated', decision=decision, heldout_recovery_gain=heldout_gain)
        state['status'] = 'pilot_checkpoint' if decision == 'continue' else 'diagnose'
        state['next_action'] = ('Review landing recovery evidence before another bounded block.'
                                if decision == 'continue' else 'Diagnose frozen landing experiment; do not expand unchanged recipe.')
        atomic_json(state_path, state)
        print(json.dumps(dict(decision=decision, heldout_recovery_gain=heldout_gain,
            success={s:m['success_rate'] for s,m in metrics.items()},
            heldout_recovery=heldout['success_rate'], completed_updates=state['completed_updates']), indent=2))
    except Exception as exc:
        attempt.update(status='failed', error=str(exc))
        state.update(status='diagnose', error=str(exc))
        atomic_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
