#!/usr/bin/env python3
"""Evidence-gated Jump-V5 nominal height continuation supervisor."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from mjlab_microduck.jump_artifacts import atomic_json, snapshot
from mjlab_microduck.jump_curriculum import (
    V5_HEIGHTS,
    V5_MAX_UPDATES,
    v5_metrics_pass,
)
from mjlab_microduck.jump_v5 import (
    add_baseline,
    finish_block,
    initial_state,
    plan_block,
    target_delta,
    terminal,
    weights,
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(command, log):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        return subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False
        ).returncode


def evaluate(root, checkpoint, seed=123, standing=False, trace=False, device="cuda:0"):
    checkpoint = Path(checkpoint)
    kind = "standing" if standing else "jump"
    name = f"{checkpoint.parent.name}_{seed}_{kind}"
    output = root / "evaluations" / f"{name}.json"
    if output.exists():
        cached = json.loads(output.read_text())
        expected_contract = 3 if cached.get("required_delta") is not None else 2
        if (cached.get("evaluator_contract") == expected_contract
                and cached.get("nominalized") is True
                and cached.get("checkpoint_sha256") == sha256(checkpoint)):
            return cached
    command = [
        sys.executable, "scripts/evaluate_jump.py",
        "--checkpoint-file", str(checkpoint),
        "--episodes", "100",
        "--seed", str(seed),
        "--device", device,
        "--json-out", str(output),
    ]
    if standing:
        command.append("--standing")
    if trace:
        command += ["--trace-out", str(output.with_suffix(".npz"))]
    if run(command, output.with_suffix(".log")):
        raise RuntimeError(f"evaluation failed: {output}")
    result = json.loads(output.read_text())
    if result["checkpoint_sha256"] != sha256(checkpoint):
        raise RuntimeError("checkpoint changed during evaluation")
    if result.get("required_delta") is not None:
        if abs(result["required_delta"] - result["target_delta"]) > 1e-9:
            raise RuntimeError("V5 evaluation did not use the exact height gate")
    return result


def run_preflight(root, state, checkpoint, device):
    command = [
        sys.executable, "-c",
        "import scipy; import scipy.optimize; print(scipy.__version__)",
    ]
    for index in range(3):
        if run(command, root / f"scipy_import_{index + 1}.log"):
            raise RuntimeError("SciPy import preflight failed")

    if state.get("smoke_passed"):
        return
    smoke = root / "smoke"
    if smoke.exists():
        result_path = smoke / "result.json"
        if not result_path.exists():
            raise RuntimeError("incomplete existing smoke directory")
        result = json.loads(result_path.read_text())
    else:
        command = [
            sys.executable, "scripts/train_jump_transfer.py",
            "--v5", "--resume", str(checkpoint),
            "--output", str(smoke),
            "--num-envs", "64",
            "--max-iterations", "5",
            "--stage", "0",
            "--target-delta", ".030",
            "--required-delta", ".030",
            "--height-weight", "1.0",
            "--heading-weight", "-.20",
            "--yaw-rate-weight", "-.02",
            "--head-bias-weight", ".5",
            "--device", device,
            "--seed", "42",
        ]
        if run(command, root / "smoke.log"):
            raise RuntimeError("V5 full-restore smoke failed")
        result = json.loads((smoke / "result.json").read_text())
    if result.get("block_completed_updates") != 5:
        raise RuntimeError("smoke did not complete exactly five updates")
    state["smoke_passed"] = True
    atomic_json(root / "state.json", state)


def train_block(root, state, source_snapshot, count, num_envs, device):
    target = target_delta(state)
    configured = weights(state)
    end = state["new_updates"] + count
    block = root / f"block_{end:04d}_h{int(round(target * 1000)):02d}"
    result_path = block / "result.json"
    if block.exists():
        if not result_path.exists():
            raise RuntimeError(f"worker output exists without result: {block}")
        result = json.loads(result_path.read_text())
    else:
        command = [
            sys.executable, "scripts/train_jump_transfer.py",
            "--v5", "--resume", str(state["checkpoint"]),
            "--output", str(block),
            "--num-envs", str(num_envs),
            "--max-iterations", str(count),
            "--stage", str(state["height_index"]),
            "--target-delta", str(target),
            "--required-delta", str(target),
            "--height-weight", str(configured["height_weight"]),
            "--heading-weight", str(configured["heading_weight"]),
            "--yaw-rate-weight", str(configured["yaw_rate_weight"]),
            "--head-bias-weight", str(configured["head_bias_weight"]),
            "--device", device,
            "--seed", "42",
            "--snapshot", source_snapshot,
        ]
        if run(command, block.with_suffix(".log")):
            state["status"] = "worker_failed"
            atomic_json(root / "state.json", state)
            raise RuntimeError(f"worker failed: {block}")
        result = json.loads(result_path.read_text())
    if result.get("block_completed_updates") != count:
        raise RuntimeError("worker update count does not match the budgeted block")
    return Path(result["checkpoint"])


def final_multiseed(root, state, device):
    checkpoint = Path(state["checkpoint"])
    results = []
    passed = True
    for seed in (123, 124, 125):
        jump = evaluate(root, checkpoint, seed, False, True, device)
        standing = evaluate(root, checkpoint, seed, True, False, device)
        seed_passed = v5_metrics_pass(jump, standing, V5_HEIGHTS[-1])
        passed &= seed_passed
        results.append({
            "seed": seed, "passed": seed_passed,
            "jump": jump, "standing": standing,
        })
    atomic_json(root / "final_multiseed.json", {
        "checkpoint": str(checkpoint), "passed": passed, "results": results,
    })
    return passed


def deliver(root, state, device):
    output = root / "delivery"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(state["checkpoint"])
    onnx = output / "policy.onnx"
    command = [
        sys.executable, "scripts/export.py", "Mjlab-Jump-Flat-MicroDuck",
        "--checkpoint-file", str(checkpoint),
        "--onnx-file", str(onnx),
        "--num-envs", "1",
        "--device", device,
    ]
    if run(command, output / "export.log"):
        raise RuntimeError("standard ONNX export failed")
    command = [
        sys.executable, "scripts/verify_jump_export.py",
        "--checkpoint", str(checkpoint),
        "--onnx", str(onnx),
        "--output", str(output / "verification.json"),
    ]
    if run(command, output / "verification.log"):
        raise RuntimeError("ONNX parity or metadata verification failed")

    cpu_json = output / "cpu_bam_30s.json"
    command = [
        sys.executable, "scripts/infer_policy.py",
        "--jump", str(onnx),
        "--headless", "--seconds", "30",
        "--jump-at", "10", "16", "22",
        "--json-out", str(cpu_json),
    ]
    cpu_code = run(command, output / "cpu_bam_30s.log")
    replay = json.loads(cpu_json.read_text()) if cpu_code == 0 and cpu_json.exists() else {}
    cpu_passed = (
        replay.get("successful_requests") == 3
        and len(replay.get("presses", [])) == 3
        and all(item.get("accepted") for item in replay.get("presses", []))
        and replay.get("initial_standing_passed") is True
        and not replay.get("invalid", True)
        and not replay.get("timeout", True)
        and replay.get("max_tilt_deg", float("inf")) <= 15.0
        and replay.get("required_delta") == V5_HEIGHTS[-1]
    )
    if not cpu_passed:
        state["status"] = "cpu_transfer_failed"
        atomic_json(root / "state.json", state)
        atomic_json(output / "manifest.json", {
            "status": state["status"], "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint), "cpu": replay,
        })
        return False

    video = output / "cpu_bam_uncut.mp4"
    video_json = output / "cpu_bam_video.json"
    video_command = command[:-2] + [
        "--video-out", str(video), "--json-out", str(video_json),
    ]
    if run(video_command, output / "cpu_bam_video.log"):
        raise RuntimeError("video generation failed after CPU gate")
    check_command = [
        sys.executable, "scripts/check_jump_video.py",
        str(video), str(output / "video_check.json"),
    ]
    decoded = run(check_command, output / "video_check.log") == 0
    state["status"] = "passed" if decoded else "video_generation_failed"
    atomic_json(root / "state.json", state)
    atomic_json(output / "manifest.json", {
        "status": state["status"],
        "visual_review_pending": decoded,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "onnx": str(onnx),
        "onnx_sha256": sha256(onnx),
        "protocol": 2,
        "policy_version": 5,
        "target_delta": V5_HEIGHTS[-1],
        "required_delta": V5_HEIGHTS[-1],
        "cpu_three_jumps": replay["successful_requests"],
        "cpu_max_tilt_deg": replay["max_tilt_deg"],
        "video_decoded": decoded,
    })
    return decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/jump-v5")
    )
    parser.add_argument(
        "--init-checkpoint", type=Path,
        default=Path("artifacts/jump-v4/run/block_0750_h30_p4/checkpoint_3000.pt"),
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-updates", type=int, default=V5_MAX_UPDATES)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument(
        "--restart-quality", action="store_true",
        help="Explicitly continue a planned quality-diagnosis stop from its checkpoint; "
             "all prior updates remain counted.",
    )
    args = parser.parse_args()
    if not 1 <= args.max_new_updates <= V5_MAX_UPDATES:
        parser.error(f"--max-new-updates must be in [1, {V5_MAX_UPDATES}]")
    if args.num_envs != 4096:
        parser.error("V5 formal training requires exactly 4096 environments")

    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else initial_state(str(args.init_checkpoint))
    )
    if args.restart_quality and state.get("status") in {
        "quality_diagnosis_heading", "quality_diagnosis_slice_budget",
        "quality_diagnosis_no_improvement",
    }:
        # This is an explicit user-requested continuation of a planned stop.
        # Keep new_updates (the global 5000-update budget) and the selected
        # checkpoint; reset only the per-slice gate so another diagnostic
        # window can collect evidence without pretending prior work vanished.
        state["quality_restarts"] = int(state.get("quality_restarts", 0)) + 1
        state["quality_restart_from"] = state["status"]
        state["slice_updates"] = 0
        state["streak"] = 0
        state["no_improvement"] = 0
        state["status"] = "quality_restart_requested"
    checkpoint = Path(state["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    source_snapshot = snapshot(root / "source")
    state["snapshot"] = source_snapshot
    atomic_json(state_path, state)
    if not args.skip_smoke:
        run_preflight(root, state, checkpoint, args.device)

    if not state["evaluations"]:
        jump = evaluate(root, checkpoint, 123, False, True, args.device)
        standing = evaluate(root, checkpoint, 123, True, False, args.device)
        add_baseline(state, jump, standing)
        atomic_json(state_path, state)

    while state["new_updates"] < args.max_new_updates and not terminal(state):
        count = plan_block(state, args.max_new_updates)
        atomic_json(state_path, state)
        if count == 0:
            break
        candidate = train_block(
            root, state, source_snapshot, count, args.num_envs, args.device
        )
        jump = evaluate(root, candidate, 123, False, True, args.device)
        standing = evaluate(root, candidate, 123, True, False, args.device)
        finish_block(state, str(candidate), count, jump, standing)
        atomic_json(state_path, state)

    if state["new_updates"] >= args.max_new_updates and not terminal(state):
        state["status"] = "budget_exhausted"
        atomic_json(state_path, state)

    if state.get("status") == "candidate_passed":
        if final_multiseed(root, state, args.device):
            deliver(root, state, args.device)
        else:
            state["status"] = "final_multiseed_failed"
            atomic_json(state_path, state)


if __name__ == "__main__":
    main()
