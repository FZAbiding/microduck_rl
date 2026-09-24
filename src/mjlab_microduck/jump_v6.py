"""Resumable evidence gate for the Jump-V6 precision-height curriculum."""
from __future__ import annotations

from mjlab_microduck.jump_curriculum import (
    V6_BLOCK_UPDATES,
    V6_HEIGHT_CAPS,
    V6_HEIGHTS,
    V6_MAX_UPDATES,
    v6_metrics_pass,
)


TERMINAL = {"passed", "budget_exhausted", "quality_diagnosis", "worker_failed"}


def initial_state(checkpoint: str) -> dict:
    return {
        "format": "microduck-jump-v6-supervisor",
        "initial_checkpoint": str(checkpoint),
        "checkpoint": str(checkpoint),
        "new_updates": 0,
        "height_index": 0,
        "slice_updates": 0,
        "streak": 0,
        "status": "initialized",
        "evaluations": [],
        "smoke_passed": False,
    }


def target_delta(state: dict) -> float:
    return V6_HEIGHTS[state["height_index"]]


def terminal(state: dict) -> bool:
    return state.get("status") in TERMINAL or state["new_updates"] >= V6_MAX_UPDATES


def plan_block(state: dict) -> int:
    if terminal(state):
        return 0
    remaining = V6_MAX_UPDATES - state["new_updates"]
    cap = V6_HEIGHT_CAPS[state["height_index"]]
    if cap is not None:
        remaining = min(remaining, cap - state["slice_updates"])
    if remaining <= 0:
        state["status"] = "quality_diagnosis"
        return 0
    preferred = 125 if state["new_updates"] < 250 else V6_BLOCK_UPDATES
    return min(preferred, remaining)


def add_baseline(state: dict, jump: dict, standing: dict) -> None:
    if state["evaluations"]:
        return
    state["evaluations"].append({
        "baseline": True,
        "checkpoint": state["checkpoint"],
        "target_delta": target_delta(state),
        "jump": jump,
        "standing": standing,
    })
    state["status"] = "continue"


def finish_block(state: dict, checkpoint: str, count: int,
                 jump: dict, standing: dict) -> str:
    if count <= 0 or count > V6_BLOCK_UPDATES:
        raise ValueError("invalid V6 block size")
    if state["new_updates"] + count > V6_MAX_UPDATES:
        raise ValueError("V6 global budget exceeded")
    state["new_updates"] += count
    state["slice_updates"] += count
    state["checkpoint"] = str(checkpoint)
    passed = v6_metrics_pass(jump, standing, target_delta(state))
    state["streak"] = state["streak"] + 1 if passed else 0
    state["evaluations"].append({
        "baseline": False,
        "checkpoint": str(checkpoint),
        "new_updates": state["new_updates"],
        "height_index": state["height_index"],
        "target_delta": target_delta(state),
        "passed": passed,
        "jump": jump,
        "standing": standing,
    })
    if passed and state["streak"] >= 2:
        if state["height_index"] == len(V6_HEIGHTS) - 1:
            state["status"] = "passed"
            return state["status"]
        state["height_index"] += 1
        state["slice_updates"] = 0
        state["streak"] = 0
        state["status"] = "advanced"
        return state["status"]
    cap = V6_HEIGHT_CAPS[state["height_index"]]
    if cap is not None and state["slice_updates"] >= cap:
        state["status"] = "quality_diagnosis"
    elif state["new_updates"] >= V6_MAX_UPDATES:
        state["status"] = "budget_exhausted"
    else:
        state["status"] = "continue"
    return state["status"]
