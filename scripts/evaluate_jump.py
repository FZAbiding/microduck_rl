#!/usr/bin/env python3
"""Evaluate full jump episodes using raw states and actual environment managers."""
import argparse
import json
from pathlib import Path
from mjlab_microduck.jump_evaluation import evaluate_checkpoint

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-file',required=True)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--mode',choices=['nominal','dr'],default='nominal')
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--json-out',type=Path,required=True)
    p.add_argument('--trace-out')
    p.add_argument('--video-out')
    p.add_argument('--standing',action='store_true')
    p.add_argument('--sequence', choices=['single', 'triple'], default=None,
                   help='V7 request schedule; defaults to single')
    p.add_argument('--stage',type=int,default=None,
                   help='DR curriculum stage; defaults to the checkpoint state')
    a=p.parse_args()
    r=evaluate_checkpoint(a.checkpoint_file,a.episodes,a.mode,a.seed,a.device,
                          a.trace_out,a.video_out,a.stage,standing=a.standing,
                          sequence=a.sequence)
    a.json_out.parent.mkdir(parents=True,exist_ok=True)
    a.json_out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='records'},indent=2))
