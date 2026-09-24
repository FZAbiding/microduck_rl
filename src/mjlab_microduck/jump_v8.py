"""Resumable two-branch supervisor state for Jump-V8."""
from __future__ import annotations

from mjlab_microduck.jump_curriculum import (
    V8_BLOCK_UPDATES, V8_BRANCH_UPDATES, V8_MAX_UPDATES,
    v8_branch_key, v8_intermediate_pass, v8_metrics_pass,
)

TERMINAL = {"passed", "budget_exhausted", "worker_failed", "final_multiseed_failed",
            "cpu_rehearsal_failed"}

def initial_state(branch_a_checkpoint: str, branch_b_checkpoint: str) -> dict:
    branches = {}
    for name, checkpoint in (("A", branch_a_checkpoint), ("B", branch_b_checkpoint)):
        branches[name] = {
            "initial_checkpoint": str(checkpoint), "checkpoint": str(checkpoint),
            "updates": 0, "evaluations": [],
        }
    return {
        "format": "microduck-jump-v8-supervisor", "new_updates": 0,
        "selected_branch": None, "checkpoint": None, "global_best": None,
        "global_best_key": None, "streak": 0, "status": "initialized",
        "smoke_passed": False, "smoke_updates": 0, "learning_rate": 1e-4,
        "consolidated": False, "branches": branches, "evaluations": [],
    }

def record_smoke(state: dict, count: int = 5) -> None:
    if state.get("smoke_passed"):
        return
    if count != 5 or state["new_updates"] + count > V8_MAX_UPDATES:
        raise ValueError("V8 smoke must contain exactly five budgeted updates")
    state.update(smoke_passed=True, smoke_updates=count, status="smoke_passed")
    state["new_updates"] += count

def terminal(state: dict) -> bool:
    return state.get("status") in TERMINAL or state["new_updates"] >= V8_MAX_UPDATES

def plan_block(state: dict) -> tuple[str | None, int]:
    if terminal(state):
        return None, 0
    remaining = V8_MAX_UPDATES - state["new_updates"]
    if state["selected_branch"] is None:
        for name in ("A", "B"):
            left = V8_BRANCH_UPDATES - state["branches"][name]["updates"]
            if left > 0:
                return name, min(V8_BLOCK_UPDATES, left, remaining)
        select_branch(state)
    return state["selected_branch"], min(V8_BLOCK_UPDATES, remaining)

def select_branch(state: dict) -> str:
    if state["selected_branch"] is not None:
        return state["selected_branch"]
    if any(state["branches"][n]["updates"] < V8_BRANCH_UPDATES for n in ("A", "B")):
        raise ValueError("both V8 comparison branches require 500 updates")
    name = max(("A", "B"), key=lambda n: state["branches"][n]["evaluations"][-1]["key"])
    state["selected_branch"] = name
    state["checkpoint"] = state["branches"][name]["checkpoint"]
    state["global_best"] = state["checkpoint"]
    state["global_best_key"] = state["branches"][name]["evaluations"][-1]["key"]
    state["status"] = "branch_selected"
    return name

def _bad_regression(state, single, triple):
    best = state.get("global_best_metrics")
    if not best:
        return False
    previous = best["triple"].get("sequence_success_rate", 0.)
    current = triple.get("sequence_success_rate", 0.)
    return (not v8_intermediate_pass(single, triple)
            or current <= previous - .15)

def finish_block(state: dict, branch_name: str, checkpoint: str, count: int,
                 single: dict, triple: dict) -> str:
    if branch_name not in state["branches"]:
        raise ValueError("unknown V8 branch")
    if count <= 0 or count > V8_BLOCK_UPDATES:
        raise ValueError("V8 block must contain 1..250 updates")
    if state["new_updates"] + count > V8_MAX_UPDATES:
        raise ValueError("V8 global update budget exceeded")
    if state["selected_branch"] is not None and branch_name != state["selected_branch"]:
        raise ValueError("only selected V8 branch may continue")
    branch = state["branches"][branch_name]
    if state["selected_branch"] is None and branch["updates"] + count > V8_BRANCH_UPDATES:
        raise ValueError("V8 comparison branch exceeds 500 updates")
    state["new_updates"] += count
    branch["updates"] += count
    key = v8_branch_key(single, triple)
    row = {"branch": branch_name, "checkpoint": str(checkpoint),
           "block_updates": count, "branch_updates": branch["updates"],
           "new_updates": state["new_updates"], "single": single, "triple": triple,
           "passed": v8_metrics_pass(single, triple), "key": key}
    branch["evaluations"].append(row)
    state["evaluations"].append(row)
    if state["global_best"] is None or key > tuple(state["global_best_key"]):
        state["global_best"] = str(checkpoint)
        state["global_best_key"] = key
        state["global_best_metrics"] = {"single": single, "triple": triple}
    if state["selected_branch"] is None:
        branch["checkpoint"] = str(checkpoint)
        if all(state["branches"][n]["updates"] >= V8_BRANCH_UPDATES for n in ("A", "B")):
            select_branch(state)
        else:
            state["status"] = "compare_branches"
        return state["status"]
    if _bad_regression(state, single, triple):
        state["checkpoint"] = state["global_best"]
        state["learning_rate"] = 5e-5
        state["streak"] = 0
        state["status"] = "rollback_best"
        return state["status"]
    state["checkpoint"] = str(checkpoint)
    passed = v8_metrics_pass(single, triple)
    state["streak"] = state["streak"] + 1 if passed else 0
    if passed and state["streak"] >= 2:
        state["consolidated"] = True
        state["status"] = "candidate_passed"
    elif state["new_updates"] >= V8_MAX_UPDATES:
        state["status"] = "budget_exhausted"
        state["checkpoint"] = state["global_best"]
    else:
        state["status"] = "continue"
    return state["status"]
