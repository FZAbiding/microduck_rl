#!/usr/bin/env python3
"""Evidence-gated Jump-v4 continuation supervisor.

The supervisor owns the 3 cm posture repair, 3.5 cm and 4 cm slices. Every
worker is at most 250 PPO updates and every checkpoint is evaluated on the fixed
100-request/100-standing samples before the next slice is selected.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

from mjlab_microduck.jump_curriculum import V4_MAX_UPDATES, V4_BLOCK_UPDATES, v4_posture_schedule


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def run(cmd, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        return subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                              env=env, check=False).returncode


def evaluate(root, checkpoint, seed=123, standing=False, trace=False, device='cuda:0'):
    name = f"{Path(checkpoint).parent.name}_{seed}_{'standing' if standing else 'jump'}"
    out = root / 'evaluations' / f'{name}.json'
    if out.exists():
        cached = json.loads(out.read_text())
        if cached.get('evaluator_contract') == 2 and cached.get('nominalized') is True:
            return cached
    cmd = [sys.executable, 'scripts/evaluate_jump.py', '--checkpoint-file', str(checkpoint),
           '--episodes', '100', '--seed', str(seed), '--device', device, '--json-out', str(out)]
    if standing:
        cmd.append('--standing')
    if trace:
        cmd += ['--trace-out', str(out.with_suffix('.npz'))]
    if run(cmd, out.with_suffix('.log')) != 0:
        raise RuntimeError(f'evaluation failed: {out}')
    result = json.loads(out.read_text())
    if result['checkpoint_sha256'] != sha256(checkpoint):
        raise RuntimeError('checkpoint changed during evaluation')
    return result


def accepted_pass(jump, standing, height_threshold, full=True):
    if standing['standing_rate'] < .95:
        return False
    if jump['success_rate'] < .95 or jump['height_rate'] < .95 or jump['takeoff_rate'] < .95:
        return False
    if full and (jump.get('heading_change_p90_deg', 180.) > 10. or
                 jump.get('heading_change_max_deg', 180.) > 20. or
                 jump.get('max_heading_error_p90_deg', jump.get('heading_error_p90_deg', 180.)) > 10. or
                 jump.get('max_heading_error_max_deg', 180.) > 20. or
                 jump.get('head_dynamic_p90_deg', 180.) > 12. or
                 jump.get('head_complete_p90_deg', 180.) > 8. or
                 jump.get('drift_p90_m', 1.) > .04):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('artifacts/jump-v4'))
    ap.add_argument('--init-checkpoint', type=Path, default=Path('artifacts/jump-v3/010_A_s2/checkpoint_2250.pt'))
    ap.add_argument('--num-envs', type=int, default=4096)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--skip-smoke', action='store_true')
    ap.add_argument('--max-new-updates', type=int, default=V4_MAX_UPDATES)
    args = ap.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        'format': 'microduck-jump-v4-supervisor', 'checkpoint': str(args.init_checkpoint),
        'new_updates': 0, 'height_slice': 0, 'posture_stage': 0, 'evaluations': [],
        'streak': 0, 'status': 'starting', 'seed': 42,
    }
    checkpoint = Path(state['checkpoint'])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    # Three explicit import checks make the former SciPy failure observable;
    # they must pass before even the 64-env restore smoke can allocate CUDA.
    import_cmd = [sys.executable, '-c', 'import scipy; import scipy.optimize; print(scipy.__version__)']
    for i in range(3):
        if run(import_cmd, root / f'scipy_import_{i+1}.log') != 0:
            raise RuntimeError('SciPy import preflight failed')
    if not args.skip_smoke and not state.get('smoke_passed'):
        smoke = root / 'smoke'
        cmd = [sys.executable, 'scripts/train_jump_transfer.py', '--v4', '--resume', str(checkpoint),
               '--output', str(smoke), '--num-envs', '64', '--max-iterations', '5', '--stage', '0',
               '--target-delta', '.030', '--heading-weight', '-.10', '--yaw-rate-weight', '-.01',
               '--head-bias-weight', '0', '--posture-stage', '0', '--device', args.device, '--seed', '42']
        if run(cmd, root / 'smoke.log') != 0:
            raise RuntimeError('v4 restore smoke failed')
        state['smoke_passed'] = True
        write_json(state_path, state)
    while state['new_updates'] < min(args.max_new_updates, V4_MAX_UPDATES):
        posture = int(state['posture_stage'])
        height = float((.030, .035, .040)[int(state['height_slice'])])
        hw, yw, hb = v4_posture_schedule(min(posture, 4))
        count = min(V4_BLOCK_UPDATES, min(args.max_new_updates, V4_MAX_UPDATES) - state['new_updates'])
        # Posture repair uses 125-update gates; all later gates use 250.
        if state['height_slice'] == 0 and state['new_updates'] < 500:
            count = min(count, 125)
        block = root / f"block_{state['new_updates'] + count:04d}_h{int(height*1000):02d}_p{posture}"
        result_path = block / 'result.json'
        if block.exists():
            # A process can die after a worker saved its checkpoint but before
            # the fixed-sample evaluator serialized its row. Reuse the
            # immutable completed block on restart instead of retraining it.
            if not result_path.exists():
                raise RuntimeError(f'worker output exists without result: {block}')
            result = json.loads(result_path.read_text())
        else:
            cmd = [sys.executable, 'scripts/train_jump_transfer.py', '--v4', '--resume', str(checkpoint),
                   '--output', str(block), '--num-envs', str(args.num_envs), '--max-iterations', str(count),
                   '--stage', str(state['height_slice']), '--target-delta', str(height),
                   '--heading-weight', str(hw), '--yaw-rate-weight', str(yw), '--head-bias-weight', str(hb),
                   '--posture-stage', str(posture), '--device', args.device, '--seed', '42']
            if run(cmd, block.with_suffix('.log')) != 0:
                state['status'] = 'worker_failed'; write_json(state_path, state); raise RuntimeError(block)
            result = json.loads(result_path.read_text())
        checkpoint = Path(result['checkpoint'])
        state['new_updates'] += count
        state['checkpoint'] = str(checkpoint)
        # Fixed sample at every boundary; standing is always evaluated too.
        jump = evaluate(root, checkpoint, 123, False, trace=True, device=args.device)
        standing = evaluate(root, checkpoint, 123, True, device=args.device)
        row = {'new_updates': state['new_updates'], 'height_slice': state['height_slice'],
               'posture_stage': posture, 'jump': jump, 'standing': standing,
               'checkpoint': str(checkpoint)}
        state['evaluations'].append(row)
        # A >5 percentage-point takeoff/height regression rolls back only the
        # just-introduced reward tier; the previous checkpoint remains immutable.
        prior = state['evaluations'][-2] if len(state['evaluations']) > 1 else None
        if prior and (jump['takeoff_rate'] < prior['jump']['takeoff_rate'] - .05 or
                      jump['height_rate'] < prior['jump']['height_rate'] - .05):
            checkpoint = Path(prior['checkpoint'])
            state['checkpoint'] = str(checkpoint)
            state['posture_stage'] = max(0, posture - 1)
            state['streak'] = 0
            state['status'] = 'rollback_reward_tier'
        elif state['height_slice'] == 0 and posture < 4 and state['new_updates'] < 500:
            state['posture_stage'] = posture + 1
            state['status'] = 'advance_posture_tier'
        elif state['height_slice'] == 0 and state['new_updates'] >= 500:
            # Require three consecutive final-posture points; if head remains
            # outside its gate at 1.0, add the 1.5 tier and restart the count.
            if accepted_pass(jump, standing, .025, full=True):
                state['streak'] += 1
            elif jump.get('head_dynamic_p90_deg', 180.) > 12. and posture < 4 and jump['takeoff_rate'] >= .90:
                state['posture_stage'] = 4
                state['streak'] = 0
            elif posture >= 4 and (jump.get('head_dynamic_p90_deg', 180.) > 12. or
                                   jump.get('head_complete_p90_deg', 180.) > 8.):
                # The maximum head-bias tier is a diagnostic boundary, not a
                # reason to raise the jump target. Three fixed samples here
                # prove that more height would only hide an unresolved pose.
                state['streak'] += 1
                if state['streak'] >= 3:
                    state['status'] = 'quality_diagnosis_failed'
                    write_json(state_path, state)
                    break
            else:
                state['streak'] = 0
            if state['streak'] >= 3:
                state['height_slice'] = 1; state['streak'] = 0; state['posture_stage'] = 4
                state['status'] = 'advance_3p5cm'
        else:
            if accepted_pass(jump, standing, .030 if state['height_slice'] == 1 else .035, full=True):
                state['streak'] += 1
            else:
                state['streak'] = 0
            if state['streak'] >= 3 and state['height_slice'] < 2:
                state['height_slice'] += 1; state['streak'] = 0
                state['status'] = 'advance_height'
            elif state['streak'] >= 3 and state['height_slice'] == 2:
                state['status'] = 'passed'; write_json(state_path, state); break
        write_json(state_path, state)
    if state.get('status') == 'passed':
        # Final multi-seed fixed samples and deployment gates are explicit,
        # serialized artifacts, and never alter the selected checkpoint.
        final = []
        for seed in (123, 124, 125):
            final.append({'seed': seed, 'jump': evaluate(root, checkpoint, seed, False, trace=True, device=args.device),
                          'standing': evaluate(root, checkpoint, seed, True, device=args.device)})
        write_json(root / 'final_multiseed.json', {'checkpoint': str(checkpoint), 'results': final})
    write_json(state_path, state)


if __name__ == '__main__':
    main()
