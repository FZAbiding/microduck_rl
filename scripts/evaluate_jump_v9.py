#!/usr/bin/env python3
"""Strict physical V9 evaluation, including V7 source checkpoints."""
import argparse
import json
from mjlab_microduck.jump_artifacts import atomic_json
from mjlab_microduck.jump_v9_evaluation import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--sequence', choices=('standing', 'recovery', 'single', 'triple'), default='single')
    p.add_argument('--json-out', required=True)
    p.add_argument('--trace-out')
    p.add_argument('--video-out')
    p.add_argument('--landing-bank-out', help='Collect safe post-landing physical states during single evaluation')
    p.add_argument('--recovery-bank', help='Use a physical landing bank for recovery evaluation')
    p.add_argument('--long-interval', action='store_true')
    a = p.parse_args()
    result = evaluate(a.checkpoint, a.episodes, a.seed, a.device, a.sequence,
                      a.trace_out, a.video_out, a.long_interval, a.landing_bank_out, a.recovery_bank)
    atomic_json(a.json_out, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
