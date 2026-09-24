"""Budgeted two-branch supervisor state for Jump-V7 recovery training."""
from __future__ import annotations

from mjlab_microduck.jump_curriculum import (
    V7_BLOCK_UPDATES,
    V7_BRANCH_UPDATES,
    V7_MAX_UPDATES,
    v7_branch_key,
    v7_metrics_pass,
)


TERMINAL = {
    "candidate_passed",
    "passed",
    "budget_exhausted",
    "worker_failed",
    "final_multiseed_failed",
    "cpu_rehearsal_failed",
}


def initial_state(v5_checkpoint: str, v6_checkpoint: str) -> dict:
    return {
        "format": "microduck-jump-v7-supervisor",
        "new_updates": 0,
        "selected_branch": None,
        "checkpoint": None,
        "streak": 0,
        "status": "initialized",
        "smoke_passed": False,
        "smoke_updates": 0,
        "branches": {
            "v5": {
                "initial_checkpoint": str(v5_checkpoint),
                "checkpoint": str(v5_checkpoint),
                "updates": 0,
                "evaluations": [],
            },
            "v6": {
                "initial_checkpoint": str(v6_checkpoint),
                "checkpoint": str(v6_checkpoint),
                "updates": 0,
                "evaluations": [],
            },
        },
        "evaluations": [],
    }


def record_smoke(state: dict, count: int = 5) -> None:
    """Charge the mandatory training smoke test to the unified V7 ledger."""
    if state.get("smoke_passed"):
        return
    if count != 5 or state["new_updates"] + count > V7_MAX_UPDATES:
        raise ValueError("V7 smoke must contain exactly five budgeted updates")
    state["smoke_passed"] = True
    state["smoke_updates"] = count
    state["new_updates"] += count
    state["status"] = "smoke_passed"


def terminal(state: dict) -> bool:
    return state.get("status") in TERMINAL or state["new_updates"] >= V7_MAX_UPDATES


def plan_block(state: dict) -> tuple[str | None, int]:
    """Return the next branch and block size without exceeding the shared budget."""
    if terminal(state):
        return None, 0
    remaining = V7_MAX_UPDATES - state["new_updates"]
    if state["selected_branch"] is None:
        for name in ("v5", "v6"):
            branch_remaining = V7_BRANCH_UPDATES - state["branches"][name]["updates"]
            if branch_remaining > 0:
                return name, min(V7_BLOCK_UPDATES, branch_remaining, remaining)
        _select_branch(state)
    return state["selected_branch"], min(V7_BLOCK_UPDATES, remaining)


def _select_branch(state: dict) -> str:
    if state["selected_branch"] is not None:
        return state["selected_branch"]
    if any(state["branches"][name]["updates"] < V7_BRANCH_UPDATES for name in ("v5", "v6")):
        raise ValueError("both V7 source branches require 500 updates before selection")
    selected = max(
        ("v5", "v6"),
        key=lambda name: v7_branch_key(
            state["branches"][name]["evaluations"][-1]["single"],
            state["branches"][name]["evaluations"][-1]["triple"],
        ),
    )
    state["selected_branch"] = selected
    state["checkpoint"] = state["branches"][selected]["checkpoint"]
    state["status"] = "branch_selected"
    return selected


def finish_block(state: dict, branch_name: str, checkpoint: str, count: int,
                 single: dict, triple: dict) -> str:
    if branch_name not in state["branches"]:
        raise ValueError("unknown V7 branch")
    if count <= 0 or count > V7_BLOCK_UPDATES:
        raise ValueError("V7 block must contain 1..250 updates")
    if state["new_updates"] + count > V7_MAX_UPDATES:
        raise ValueError("V7 global update budget exceeded")
    if state["selected_branch"] is not None and branch_name != state["selected_branch"]:
        raise ValueError("only the selected V7 branch may continue")

    branch = state["branches"][branch_name]
    if state["selected_branch"] is None and branch["updates"] + count > V7_BRANCH_UPDATES:
        raise ValueError("V7 comparison branch exceeds 500 updates")
    state["new_updates"] += count
    branch["updates"] += count
    branch["checkpoint"] = str(checkpoint)
    row = {
        "branch": branch_name,
        "checkpoint": str(checkpoint),
        "block_updates": count,
        "branch_updates": branch["updates"],
        "new_updates": state["new_updates"],
        "passed": v7_metrics_pass(single, triple),
        "single": single,
        "triple": triple,
    }
    branch["evaluations"].append(row)
    state["evaluations"].append(row)

    if state["selected_branch"] is None:
        if all(state["branches"][name]["updates"] >= V7_BRANCH_UPDATES
               for name in ("v5", "v6")):
            _select_branch(state)
        else:
            state["status"] = "compare_branches"
        # Comparison evidence does not count toward the consecutive post-
        # selection acceptance streak.
        state["streak"] = 0
        return state["status"]

    state["checkpoint"] = str(checkpoint)
    passed = v7_metrics_pass(single, triple)
    state["streak"] = state["streak"] + 1 if passed else 0
    if state["streak"] >= 2:
        state["status"] = "candidate_passed"
    elif state["new_updates"] >= V7_MAX_UPDATES:
        state["status"] = "budget_exhausted"
    else:
        state["status"] = "continue"
    return state["status"]
