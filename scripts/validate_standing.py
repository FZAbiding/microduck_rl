#!/usr/bin/env python3
import argparse
import json
from mjlab_microduck.jump_evaluation import standing_gate
if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--onnx', default='artifacts/official/alpha_stand.onnx')
    p.add_argument('--init-checkpoint', default='artifacts/jump_v2_transfer/standing_init.pt')
    p.add_argument('--json-out', default='artifacts/jump_v2_transfer/standing_gate.json')
    p.add_argument('--device', default='cuda:0')
    a=p.parse_args()
    r=standing_gate(a.onnx,a.init_checkpoint,a.json_out,a.device)
    print(json.dumps(r,indent=2))
    raise SystemExit(0 if r['passed'] else 1)
