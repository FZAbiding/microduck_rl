#!/usr/bin/env python3
"""Supervise the fixed-height Jump-V7 recovery and repeatability stage."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from mjlab_microduck.jump_artifacts import atomic_json, snapshot
from mjlab_microduck.jump_curriculum import v7_metrics_pass
from mjlab_microduck.jump_v7 import (
    finish_block,
    initial_state,
    plan_block,
    record_smoke,
    terminal,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        return subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False
        ).returncode


def evaluate(root: Path, checkpoint: Path, seed: int, sequence: str,
             device: str, trace: bool = False, video: bool = False) -> dict:
    name = f'{checkpoint.parent.name}_{seed}_{sequence}'
    output = root / 'evaluations' / f'{name}.json'
    if output.exists():
        cached = json.loads(output.read_text())
        if (cached.get('checkpoint_sha256') == sha256(checkpoint)
                and cached.get('evaluator_contract') == 7
                and cached.get('sequence') == sequence):
            return cached
    command = [
        sys.executable, 'scripts/evaluate_jump.py',
        '--checkpoint-file', str(checkpoint),
        '--episodes', '100',
        '--seed', str(seed),
        '--device', device,
        '--sequence', sequence,
        '--json-out', str(output),
    ]
    if trace:
        command += ['--trace-out', str(output.with_suffix('.npz'))]
    if video:
        command += ['--video-out', str(output.with_suffix('.mp4'))]
    if run(command, output.with_suffix('.log')):
        raise RuntimeError(f'V7 {sequence} evaluation failed: {output}')
    result = json.loads(output.read_text())
    if result.get('checkpoint_sha256') != sha256(checkpoint):
        raise RuntimeError('checkpoint changed during V7 evaluation')
    return result


def smoke(root: Path, state: dict, checkpoint: Path, device: str) -> None:
    if state.get('smoke_passed'):
        return
    output = root / 'smoke'
    result_path = output / 'result.json'
    if output.exists():
        if not result_path.exists():
            raise RuntimeError(f'incomplete existing smoke output: {output}')
    else:
        command = [
            sys.executable, 'scripts/train_jump_transfer.py',
            '--v7',
            '--resume', str(checkpoint),
            '--output', str(output),
            '--num-envs', '64',
            '--max-iterations', '5',
            '--target-delta', '.03',
            '--required-delta', '.03',
            '--device', device,
            '--seed', '42',
        ]
        if run(command, root / 'smoke.log'):
            raise RuntimeError('V7 64-env x 5-update smoke failed')
    result = json.loads(result_path.read_text())
    if result.get('block_completed_updates') != 5:
        raise RuntimeError('V7 smoke did not complete exactly five updates')
    record_smoke(state, 5)
    atomic_json(root / 'state.json', state)


def train_block(root: Path, state: dict, branch_name: str, count: int,
                source: str, num_envs: int, device: str) -> Path:
    branch = state['branches'][branch_name]
    end = state['new_updates'] + count
    branch_end = branch['updates'] + count
    output = root / f'{branch_name}_block_{branch_end:04d}_ledger_{end:04d}'
    result_path = output / 'result.json'
    if output.exists():
        if not result_path.exists():
            raise RuntimeError(f'incomplete existing worker output: {output}')
    else:
        command = [
            sys.executable, 'scripts/train_jump_transfer.py',
            '--v7',
            '--resume', str(branch['checkpoint']),
            '--output', str(output),
            '--num-envs', str(num_envs),
            '--max-iterations', str(count),
            '--target-delta', '.03',
            '--required-delta', '.03',
            '--device', device,
            '--seed', '42',
            '--snapshot', source,
        ]
        if run(command, output.with_suffix('.log')):
            state['status'] = 'worker_failed'
            atomic_json(root / 'state.json', state)
            raise RuntimeError(f'V7 worker failed: {output}')
    result = json.loads(result_path.read_text())
    if result.get('block_completed_updates') != count:
        raise RuntimeError('V7 worker update count mismatch')
    return Path(result['checkpoint'])


def final_multiseed(root: Path, checkpoint: Path, device: str) -> bool:
    rows = []
    for seed in (123, 124, 125):
        single = evaluate(root, checkpoint, seed, 'single', device, trace=True)
        triple = evaluate(root, checkpoint, seed, 'triple', device, trace=True)
        rows.append({
            'seed': seed,
            'passed': v7_metrics_pass(single, triple),
            'single': single,
            'triple': triple,
        })
    passed = all(row['passed'] for row in rows)
    atomic_json(root / 'final_multiseed.json', {
        'checkpoint': str(checkpoint),
        'passed': passed,
        'results': rows,
    })
    return passed


def export_and_rehearse(root: Path, checkpoint: Path) -> bool:
    onnx = root / 'policy.onnx'
    export_command = [
        'uv', 'run', 'scripts/export.py', 'Mjlab-Jump-Flat-MicroDuck',
        '--checkpoint-file', str(checkpoint),
        '--onnx-file', str(onnx),
        '--device', 'cpu',
        '--num-envs', '1',
    ]
    if run(export_command, root / 'export.log'):
        return False
    verify = root / 'export_verification.json'
    if run([
        sys.executable, 'scripts/verify_jump_export.py',
        '--checkpoint', str(checkpoint),
        '--onnx', str(onnx),
        '--output', str(verify),
    ], root / 'verify_export.log'):
        return False
    replay = root / 'cpu_bam_triple.json'
    if run([
        'uv', 'run', 'scripts/infer_policy.py',
        '--jump', str(onnx),
        '--headless',
        '--seconds', '30',
        '--jump-at', '7', '14', '21',
        '--json-out', str(replay),
    ], root / 'cpu_bam_triple.log'):
        return False
    result = json.loads(replay.read_text())
    passed = (
        result.get('successful_requests') == 3
        and len(result.get('presses', ())) == 3
        and all(row.get('accepted') for row in result.get('presses', ()))
        and not result.get('invalid')
        and not result.get('timeout')
        and not result.get('body_contact')
    )
    atomic_json(root / 'cpu_bam_gate.json', {
        'passed': passed,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256(checkpoint),
        'onnx': str(onnx),
        'onnx_sha256': sha256(onnx),
        'replay': result,
    })
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/jump-v7'))
    parser.add_argument('--v5-checkpoint', type=Path, default=Path(
        'artifacts/jump-v5/block_1500_h30/checkpoint_4375.pt'
    ))
    parser.add_argument('--v6-checkpoint', type=Path, default=Path(
        'artifacts/jump-v6/block_01250_h030/checkpoint_5625.pt'
    ))
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    for checkpoint in (args.v5_checkpoint, args.v6_checkpoint):
        if not checkpoint.exists():
            parser.error(f'missing source checkpoint: {checkpoint}')
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = Path('/tmp/microduck_jump_gpu0.lock')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = args.output / 'state.json'
        state = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else initial_state(str(args.v5_checkpoint), str(args.v6_checkpoint))
        )
        smoke(args.output, state, args.v5_checkpoint, args.device)
        source = snapshot(args.output / 'source')
        while not terminal(state):
            branch_name, count = plan_block(state)
            if not branch_name or count <= 0:
                break
            checkpoint = train_block(
                args.output, state, branch_name, count, source,
                args.num_envs, args.device,
            )
            single = evaluate(
                args.output, checkpoint, 123, 'single', args.device, trace=True
            )
            triple = evaluate(
                args.output, checkpoint, 123, 'triple', args.device, trace=True
            )
            finish_block(
                state, branch_name, str(checkpoint), count, single, triple
            )
            atomic_json(state_path, state)

        if state.get('status') == 'candidate_passed':
            checkpoint = Path(state['checkpoint'])
            if not final_multiseed(args.output, checkpoint, args.device):
                state['status'] = 'final_multiseed_failed'
            elif not export_and_rehearse(args.output, checkpoint):
                state['status'] = 'cpu_rehearsal_failed'
            else:
                state['status'] = 'passed'
            atomic_json(state_path, state)


if __name__ == '__main__':
    main()
