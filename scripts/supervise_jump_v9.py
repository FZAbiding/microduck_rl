#!/usr/bin/env python3
"""Bounded nominal A/B pilot. Never launch later curricula automatically."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from mjlab_microduck.jump_artifacts import atomic_json, sha256
from mjlab_microduck.jump_v9 import (
    MAX_UPDATES, blocked_jump_experience, safe_candidate, selection_key, validate_metrics, worker_arguments,
)


def read(path):
    return json.loads(Path(path).read_text())


def run(arguments, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        subprocess.run([sys.executable, *arguments], stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def evaluation(root, checkpoint, sequence, device, label, seed=123):
    output = root/'evaluations'/f'{label}-{sequence}-{seed}.json'
    if output.exists():
        raise ValueError(f'refusing stale evaluation: {output}')
    run(['scripts/evaluate_jump_v9.py', '--checkpoint', str(checkpoint),
         '--sequence', sequence, '--episodes', '100', '--seed', str(seed),
         '--json-out', str(output), '--device', device], output.with_suffix('.log'))
    result = read(output)
    validate_metrics(result)
    if result['checkpoint_sha256'] != sha256(checkpoint):
        raise RuntimeError('evaluation checkpoint mismatch')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('artifacts/jump-v9'))
    p.add_argument('--blocks-per-branch', type=int, default=5)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--branches', nargs='+', choices=('A', 'B'), default=['A', 'B'])
    p.add_argument('--idle-std-scale', type=float, default=1.)
    p.add_argument('--readiness-diagnostic', type=Path)
    p.add_argument('--smoke-result', type=Path)
    p.add_argument('--export-check', type=Path)
    a = p.parse_args()
    if not 1 <= a.blocks_per_branch <= 5:
        p.error('pilot limit is 1..5 blocks of 50 per branch')
    root, state_path = a.output, a.output/'state.json'
    if state_path.exists():
        state = read(state_path)
        if state['status'] not in ('pilot_checkpoint', 'pilot_complete', 'diagnose'):
            raise RuntimeError('unfinished/failed attempt requires diagnosis; no implicit retry')
    else:
        smoke_results = sorted(root.glob('smoke-*/result.json'))
        smoke = read(smoke_results[-1])
        export = read(root/'preflight/export-check.json')
        if not (smoke['block_completed_updates'] == 5 and export['passed']
                and smoke['checkpoint_sha256'] == export['checkpoint_sha256']):
            raise RuntimeError('V9 smoke/export gate not passed')
        smoke_updates = sum(read(path)['block_completed_updates'] for path in smoke_results)
        state = dict(status='pilot_checkpoint', budget_limit=MAX_UPDATES,
                     reserved_updates=smoke_updates, completed_updates=smoke_updates,
                     smoke_results=[str(p) for p in smoke_results], attempts=[], branches={})
        for name in ('A', 'B'):
            metrics = {seq: read(root/'baselines'/f'{name}-{seq}.json')
                       for seq in ('single', 'triple', 'standing', 'recovery')}
            for seq, value in metrics.items():
                validate_metrics(value)
                if value['episodes'] < 100 or value['seed'] != 123:
                    raise RuntimeError('incomplete baseline')
                if seq == 'single' and value['seconds'] < 20:
                    raise RuntimeError('truncated baseline')
            checkpoint = metrics['single']['checkpoint']
            if any(m['checkpoint_sha256'] != sha256(checkpoint) for m in metrics.values()):
                raise RuntimeError('baseline source mismatch')
            key = selection_key(metrics['single'], metrics['triple'])
            usable = (metrics['standing']['success_rate'] >= .99
                      and all(safe_candidate(m) for m in metrics.values()))
            state['branches'][name] = dict(checkpoint=checkpoint, best_checkpoint=checkpoint if key else None,
                best_key=key, best_metrics=metrics, learning_rate=5e-5, updates=0, rollbacks=0,
                no_progress=0, status='continue' if usable else 'diagnose')
        atomic_json(state_path, state)
    if not 0 < a.idle_std_scale <= 1:
        p.error('idle std scale must be in (0, 1]')
    # Audit old attempts before allowing a new block, not after training it.
    for name, branch in state['branches'].items():
        attempts = [t for t in state['attempts'] if t['label'].startswith(name+'-')
                    and t.get('status') == 'evaluated']
        if attempts and blocked_jump_experience(read(root/attempts[-1]['label']/'progress.json')):
            branch.update(status='diagnose', reason='no_jump_or_recovery_completion_experience')
    for name in a.branches:
        branch = state['branches'][name]
        if branch['status'] == 'diagnose' or branch.get('idle_std_scale', 1.) != a.idle_std_scale:
            if not all((a.readiness_diagnostic, a.smoke_result, a.export_check)):
                raise RuntimeError('diagnosed recipe change requires readiness diagnostic and smoke/export evidence')
            diagnostic, smoke, export = map(read, (a.readiness_diagnostic, a.smoke_result, a.export_check))
            parent_hash = sha256(branch['checkpoint'])
            matches = [r for r in diagnostic['results'] if r['idle_std_scale'] == a.idle_std_scale]
            if not (diagnostic['ppo_updates'] == 0 and diagnostic['episodes'] >= 100
                    and diagnostic['checkpoint_sha256'] == parent_hash
                    and matches and matches[0]['strict_ready_at_7s'] >= .99
                    and matches[0]['terminations'] == 0
                    and smoke['parent_sha256'] == parent_hash
                    and smoke['block_completed_updates'] == 5
                    and smoke['idle_std_scale'] == a.idle_std_scale
                    and export['passed'] and export['checkpoint_sha256'] == smoke['checkpoint_sha256']
                    and str(a.smoke_result) in state['smoke_results']):
                raise RuntimeError('readiness/smoke/export evidence mismatch')
            branch.update(status='continue', idle_std_scale=a.idle_std_scale,
                          recipe_evidence=dict(readiness=str(a.readiness_diagnostic),
                                               smoke=str(a.smoke_result), export=str(a.export_check)))
            branch.pop('reason', None)
    atomic_json(state_path, state)
    try:
        for block in range(a.blocks_per_branch):
            for name, branch in state['branches'].items():
                if name not in a.branches:
                    continue
                if branch['updates'] >= (block+1)*50 or branch['status'] == 'diagnose':
                    continue
                if state['reserved_updates']+50 > MAX_UPDATES:
                    raise RuntimeError('budget exceeded')
                label = f'{name}-pilot-{branch["updates"]+50:04d}'
                output = root/label
                arguments = worker_arguments(branch, output, 50, a.num_envs, a.device)
                parent_hash = sha256(branch['checkpoint'])
                attempt = dict(label=label, parent_checkpoint=branch['checkpoint'], parent_sha256=parent_hash,
                               learning_rate=branch['learning_rate'], idle_std_scale=branch.get('idle_std_scale', 1.),
                               reserved_updates=50, status='running')
                state['attempts'].append(attempt)
                state['reserved_updates'] += 50  # Failed/retried reservations never disappear.
                state['status'] = 'training'
                atomic_json(state_path, state)
                run(arguments, output.with_suffix('.log'))
                result = read(output/'result.json')
                if not (result['parent_sha256'] == parent_hash and result['block_completed_updates'] == 50
                        and result['actual_learning_rate'] == branch['learning_rate']
                        and result['idle_std_scale'] == branch.get('idle_std_scale', 1.)
                        and result['checkpoint_sha256'] == sha256(result['checkpoint'])):
                    raise RuntimeError('worker provenance/count/LR mismatch')
                checkpoint = result['checkpoint']
                attempt.update(status='trained', completed_updates=50, checkpoint=checkpoint)
                branch['updates'] += 50
                state['completed_updates'] += 50
                state['status'] = 'evaluating'
                atomic_json(state_path, state)
                metrics = {seq: evaluation(root, checkpoint, seq, a.device, label)
                           for seq in ('single', 'triple', 'standing', 'recovery')}
                key = selection_key(metrics['single'], metrics['triple'])
                old = branch['best_metrics']
                regression = (key is None or not all(safe_candidate(m) for m in metrics.values())
                    or metrics['standing']['success_rate'] < .99 or
                    any(metrics[s]['success_rate'] < old[s]['success_rate']-.10
                        for s in ('single', 'triple', 'recovery')))
                if regression:
                    # Fixed-seed rerun records repeatability; GPU physics is not promised bitwise deterministic.
                    confirm = {seq: evaluation(root, checkpoint, seq, a.device, label+'-confirm')
                               for seq in ('single', 'triple')}
                    attempt['confirmation'] = {s: m['success_rate'] for s, m in confirm.items()}
                    branch['rollbacks'] += 1
                    branch['checkpoint'] = branch['best_checkpoint']
                    branch['learning_rate'] = 2.5e-5
                    branch['status'] = 'diagnose' if branch['rollbacks'] >= 2 or not branch['checkpoint'] else 'rollback'
                else:
                    branch['checkpoint'] = checkpoint
                    branch['status'] = 'continue'
                    improved = branch['best_key'] is None or key > tuple(branch['best_key'])
                    recovery_improved = metrics['recovery']['success_rate'] > old['recovery']['success_rate']
                    branch['no_progress'] = 0 if improved or recovery_improved else branch['no_progress']+1
                    if improved:
                        branch.update(best_checkpoint=checkpoint, best_key=key, best_metrics=metrics)
                    if branch['no_progress'] >= 2:
                        branch['status'] = 'diagnose'
                if blocked_jump_experience(read(output/'progress.json')):
                    branch.update(status='diagnose', reason='no_jump_or_recovery_completion_experience')
                attempt.update(status='evaluated', decision=branch['status'],
                    metrics={s: {k: v for k, v in m.items() if k != 'records'} for s, m in metrics.items()})
                state['status'] = 'pilot_checkpoint'
                atomic_json(state_path, state)
                print(json.dumps(dict(branch=name, updates=branch['updates'], decision=branch['status'],
                    single=metrics['single']['success_rate'], triple=metrics['triple']['success_rate'],
                    recovery=metrics['recovery']['success_rate']), indent=2), flush=True)
        # Also review existing completed attempts when invoked for the same boundary.
        for name, branch in state['branches'].items():
            attempted = [t for t in state['attempts'] if t['label'].startswith(name+'-')
                         and t.get('status') == 'evaluated']
            if attempted and blocked_jump_experience(read(root/attempted[-1]['label']/'progress.json')):
                branch.update(status='diagnose', reason='no_jump_or_recovery_completion_experience')
        state['status'] = ('diagnose' if all(b['status'] == 'diagnose' for b in state['branches'].values())
                           else 'pilot_complete' if a.blocks_per_branch == 5 else 'pilot_checkpoint')
        atomic_json(state_path, state)
    except Exception as exc:
        state['status'], state['error'] = 'diagnose', str(exc)
        atomic_json(state_path, state)
        raise


if __name__ == '__main__':
    main()
