#!/usr/bin/env python3
"""Supervise Jump-V6 precision repair and 3 cm -> 10 cm continuation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from mjlab_microduck.jump_artifacts import atomic_json, snapshot
from mjlab_microduck.jump_curriculum import V6_HEIGHTS, v6_metrics_pass
from mjlab_microduck.jump_v6 import (
    add_baseline, finish_block, initial_state, plan_block, target_delta, terminal
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        return subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False
        ).returncode


def evaluate(root: Path, checkpoint: Path, seed: int, standing: bool,
             device: str, trace: bool = False) -> dict:
    kind = "standing" if standing else "jump"
    name = f"{checkpoint.parent.name}_{seed}_{kind}"
    output = root / "evaluations" / f"{name}.json"
    if output.exists():
        cached = json.loads(output.read_text())
        if cached.get("checkpoint_sha256") == sha256(checkpoint):
            return cached
    command = [
        sys.executable, "scripts/evaluate_jump.py",
        "--checkpoint-file", str(checkpoint),
        "--episodes", "100", "--seed", str(seed),
        "--device", device, "--json-out", str(output),
    ]
    if standing:
        command.append("--standing")
    if trace:
        command += ["--trace-out", str(output.with_suffix(".npz"))]
    if run(command, output.with_suffix(".log")):
        raise RuntimeError(f"evaluation failed: {output}")
    result = json.loads(output.read_text())
    if result.get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError("checkpoint changed during evaluation")
    return result


def preflight(root: Path) -> None:
    scipy = [sys.executable, "-c", "import scipy; import scipy.optimize"]
    for index in range(3):
        if run(scipy, root / f"scipy_import_{index + 1}.log"):
            raise RuntimeError("SciPy import preflight failed")
    smoke = root / "smoke4" / "result.json"
    if not smoke.exists():
        raise RuntimeError("V6 64-env x 5-update smoke result is missing")
    result = json.loads(smoke.read_text())
    if result.get("block_completed_updates") != 5:
        raise RuntimeError("V6 smoke did not complete exactly five updates")


def train_block(root: Path, state: dict, source: str, count: int,
                num_envs: int, device: str) -> Path:
    target = target_delta(state)
    end = state["new_updates"] + count
    block = root / f"block_{end:05d}_h{int(round(target * 1000)):03d}"
    result_path = block / "result.json"
    if block.exists():
        if not result_path.exists():
            raise RuntimeError(f"incomplete worker output: {block}")
    else:
        command = [
            sys.executable, "scripts/train_jump_transfer.py",
            "--v6", "--resume", str(state["checkpoint"]),
            "--output", str(block), "--num-envs", str(num_envs),
            "--max-iterations", str(count),
            "--stage", str(state["height_index"]),
            "--target-delta", str(target), "--required-delta", str(target),
            "--device", device, "--seed", "42", "--snapshot", source,
        ]
        if run(command, block.with_suffix(".log")):
            state["status"] = "worker_failed"
            atomic_json(root / "state.json", state)
            raise RuntimeError(f"worker failed: {block}")
    result = json.loads(result_path.read_text())
    if result.get("block_completed_updates") != count:
        raise RuntimeError("worker update count mismatch")
    return Path(result["checkpoint"])


def final_multiseed(root: Path, state: dict, device: str) -> bool:
    checkpoint = Path(state["checkpoint"])
    rows = []
    for seed in (123, 124, 125):
        jump = evaluate(root, checkpoint, seed, False, device, trace=True)
        standing = evaluate(root, checkpoint, seed, True, device)
        rows.append({
            "seed": seed,
            "passed": v6_metrics_pass(jump, standing, V6_HEIGHTS[-1]),
            "jump": jump, "standing": standing,
        })
    passed = all(row["passed"] for row in rows)
    atomic_json(root / "final_multiseed.json", {
        "checkpoint": str(checkpoint), "passed": passed, "results": rows
    })
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/jump-v6"))
    parser.add_argument("--init-checkpoint", type=Path, default=Path(
        "artifacts/jump-v5/block_1500_h30/checkpoint_4375.pt"
    ))
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = Path("/tmp/microduck_jump_gpu0.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = args.output / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
        else:
            state = initial_state(str(args.init_checkpoint))
            atomic_json(state_path, state)
        preflight(args.output)
        state["smoke_passed"] = True
        source = snapshot(args.output / "source")
        if not state["evaluations"]:
            checkpoint = Path(state["checkpoint"])
            jump = evaluate(args.output, checkpoint, 123, False, args.device, trace=True)
            standing = evaluate(args.output, checkpoint, 123, True, args.device)
            add_baseline(state, jump, standing)
            atomic_json(state_path, state)
        while not terminal(state):
            count = plan_block(state)
            if count <= 0:
                atomic_json(state_path, state)
                break
            checkpoint = train_block(
                args.output, state, source, count, args.num_envs, args.device
            )
            jump = evaluate(args.output, checkpoint, 123, False, args.device, trace=True)
            standing = evaluate(args.output, checkpoint, 123, True, args.device)
            finish_block(state, str(checkpoint), count, jump, standing)
            atomic_json(state_path, state)
        if state.get("status") == "passed":
            state["final_multiseed_passed"] = final_multiseed(
                args.output, state, args.device
            )
            if not state["final_multiseed_passed"]:
                state["status"] = "final_multiseed_failed"
            atomic_json(state_path, state)


if __name__ == "__main__":
    main()
