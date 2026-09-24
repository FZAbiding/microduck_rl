#!/usr/bin/env python3
"""Create a new actor-only training start, never an official resumed run."""
import argparse
import json
from mjlab_microduck.jump_transfer import prepare_transfer

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx', required=True)
    p.add_argument('--output', default='artifacts/jump_v2_transfer/standing_init.pt')
    args = p.parse_args()
    print(json.dumps(prepare_transfer(args.onnx, args.output), indent=2))
