#!/usr/bin/env python3
"""Supervise the nominal Jump-V8 two-branch recovery run."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from mjlab_microduck.jump_artifacts import atomic_json, snapshot
from mjlab_microduck.jump_v8 import (
    finish_block, initial_state, plan_block, record_smoke, terminal,
)

def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode

def train(root, state, branch, count, device, num_envs):
    b=state["branches"][branch]
    end=b["updates"]+count
    out=root/f"{branch}_block_{end:04d}_ledger_{state['new_updates']+count:04d}"
    result=out/"result.json"
    if not result.exists():
        cmd=[sys.executable,"scripts/train_jump_transfer.py","--v8","--resume",
             b["checkpoint"],"--output",str(out),"--num-envs",str(num_envs),
             "--max-iterations",str(count),"--target-delta",".032",
             "--required-delta",".030","--device",device,"--seed","42",
             "--snapshot",str(root/"source")]
        if run(cmd,out.with_suffix(".log")):
            raise RuntimeError(f"V8 worker failed: {out}")
    data=json.loads(result.read_text())
    if data.get("block_completed_updates") != count:
        raise RuntimeError(f"V8 worker count mismatch: {data}")
    return Path(data["checkpoint"])

def evaluate(root, checkpoint, sequence, device):
    out=root/"evaluations"/f"{checkpoint.parent.name}_{sequence}.json"
    if not out.exists():
        cmd=[sys.executable,"scripts/evaluate_jump.py","--checkpoint-file",
             str(checkpoint),"--episodes","100","--seed","123","--device",device,
             "--sequence",sequence,"--json-out",str(out)]
        if run(cmd,out.with_suffix(".log")):
            raise RuntimeError(f"V8 evaluation failed: {out}")
    return json.loads(out.read_text())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,default=Path("artifacts/jump-v8"))
    ap.add_argument("--branch-a",type=Path,default=Path(
        "artifacts/jump-v7/v5_block_2000_ledger_2505/checkpoint_6375.pt"))
    ap.add_argument("--branch-b",type=Path,default=Path(
        "artifacts/jump-v7/v5_block_1750_ledger_2255/checkpoint_6125.pt"))
    ap.add_argument("--num-envs",type=int,default=4096)
    ap.add_argument("--device",default="cuda:0")
    a=ap.parse_args()
    for path in (a.branch_a,a.branch_b):
        if not path.exists(): ap.error(f"missing checkpoint: {path}")
    a.output.mkdir(parents=True,exist_ok=True)
    state_path=a.output/"state.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else initial_state(str(a.branch_a),str(a.branch_b))
    if not state.get("smoke_passed"):
        smoke=a.output/"smoke"
        if not (smoke/"result.json").exists():
            cmd=[sys.executable,"scripts/train_jump_transfer.py","--v8","--resume",
                 str(a.branch_a),"--output",str(smoke),"--num-envs","64",
                 "--max-iterations","5","--target-delta",".032",
                 "--required-delta",".030","--device",a.device,"--seed","42",
                 "--snapshot",str(a.output/"source")]
            if run(cmd,a.output/"smoke.log"): raise RuntimeError("V8 smoke failed")
        data=json.loads((smoke/"result.json").read_text())
        if data.get("block_completed_updates") != 5: raise RuntimeError("bad V8 smoke count")
        record_smoke(state,5); atomic_json(state_path,state)
    while not terminal(state):
        branch,count=plan_block(state)
        if not branch: break
        checkpoint=train(a.output,state,branch,count,a.device,a.num_envs)
        single=evaluate(a.output,checkpoint,"single",a.device)
        triple=evaluate(a.output,checkpoint,"triple",a.device)
        finish_block(state,branch,str(checkpoint),count,single,triple)
        atomic_json(state_path,state)
        print(json.dumps({"status":state["status"],"branch":branch,
                          "new_updates":state["new_updates"],
                          "checkpoint":state.get("checkpoint")},indent=2),flush=True)
    atomic_json(a.output/"final_state.json",state)

if __name__=="__main__":
    main()
