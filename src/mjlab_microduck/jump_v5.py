"""Pure Jump-V5 curriculum state used by the resumable supervisor."""
from __future__ import annotations

from copy import deepcopy

from mjlab_microduck.jump_curriculum import (
    V5_BASE_HEADING_WEIGHT,
    V5_BASE_YAW_RATE_WEIGHT,
    V5_BLOCK_UPDATES,
    V5_HEAD_BIAS_WEIGHT,
    V5_HEIGHT_CAPS,
    V5_HEIGHTS,
    V5_MAX_UPDATES,
    V5_STRONG_HEADING_WEIGHT,
    V5_STRONG_YAW_RATE_WEIGHT,
    v5_heading_pass,
    v5_improved,
    v5_metrics_pass,
    v5_reward_regressed,
)


def initial_state(checkpoint: str) -> dict:
    return {
        "format": "microduck-jump-v5-supervisor",
        "initial_checkpoint": checkpoint,
        "checkpoint": checkpoint,
        "new_updates": 0,
        "height_index": 0,
        "slice_updates": 0,
        "bootstrap_phase": 0,
        "height_weight": 1.0,
        "heading_weight": V5_BASE_HEADING_WEIGHT,
        "yaw_rate_weight": V5_BASE_YAW_RATE_WEIGHT,
        "head_bias_weight": V5_HEAD_BIAS_WEIGHT,
        "height_weight_raised": False,
        "height_weight_raise_blocked": False,
        "heading_strengthened": False,
        "heading_strengthen_blocked": False,
        "streak": 0,
        "no_improvement": 0,
        "evaluations": [],
        "accepted_evaluations": [],
        "pending_reward_change": None,
        "status": "starting",
        "seed": 42,
    }


def target_delta(state: dict) -> float:
    return V5_HEIGHTS[int(state["height_index"])]


def weights(state: dict) -> dict:
    return {
        "height_weight": float(state["height_weight"]),
        "heading_weight": float(state["heading_weight"]),
        "yaw_rate_weight": float(state["yaw_rate_weight"]),
        "head_bias_weight": V5_HEAD_BIAS_WEIGHT,
    }


def add_baseline(state: dict, jump: dict, standing: dict) -> None:
    """Record the immutable V4 starting point for the first rollback comparison."""
    if state["evaluations"]:
        raise ValueError("baseline already recorded")
    row = {
        "baseline": True,
        "selected": True,
        "checkpoint": state["checkpoint"],
        "new_updates": 0,
        "height_index": 0,
        "target_delta": V5_HEIGHTS[0],
        "jump": jump,
        "standing": standing,
        "weights": weights(state),
    }
    state["evaluations"].append(row)
    state["last_selected_evaluation"] = deepcopy(row)
    state["status"] = "baseline_evaluated"


def _schedule_reward_change(state: dict, kind: str, **new_weights: float) -> None:
    if state.get("pending_reward_change") is not None:
        raise RuntimeError("another reward change is already pending")
    baseline = state["last_selected_evaluation"]
    state["pending_reward_change"] = {
        "kind": kind,
        "checkpoint": state["checkpoint"],
        "weights": weights(state),
        "jump": baseline["jump"],
        "standing": baseline["standing"],
    }
    state.update(new_weights)
    state["streak"] = 0


def plan_block(state: dict, max_updates: int = V5_MAX_UPDATES) -> int:
    """Prepare the next immutable worker block and return its update count."""
    hard_limit = min(int(max_updates), V5_MAX_UPDATES)
    if state["new_updates"] >= hard_limit:
        state["status"] = "budget_exhausted"
        return 0

    phase = int(state["bootstrap_phase"])
    if phase == 1 and state["height_weight"] == 1.0:
        _schedule_reward_change(state, "height_1p5", height_weight=1.5)
        state["status"] = "increase_height_reward_1p5"
    elif phase == 2:
        if (not v5_heading_pass(state["last_selected_evaluation"]["jump"])
                and not state["heading_strengthened"]
                and not state["heading_strengthen_blocked"]):
            _schedule_reward_change(
                state, "strong_heading",
                heading_weight=V5_STRONG_HEADING_WEIGHT,
                yaw_rate_weight=V5_STRONG_YAW_RATE_WEIGHT,
            )
            state["heading_strengthened"] = True
            state["status"] = "strengthen_heading"
        else:
            state["bootstrap_phase"] = 3
            state["streak"] = 0
            phase = 3

    count = 125 if int(state["bootstrap_phase"]) < 3 else V5_BLOCK_UPDATES
    cap = V5_HEIGHT_CAPS[int(state["height_index"])]
    if cap is not None:
        count = min(count, int(cap) - int(state["slice_updates"]))
    count = min(count, hard_limit - int(state["new_updates"]))
    if count <= 0:
        state["status"] = (
            "quality_diagnosis_heading"
            if int(state["height_index"]) == 0
            and not v5_heading_pass(state["last_selected_evaluation"]["jump"])
            else "quality_diagnosis_slice_budget"
        )
        return 0
    return count


def finish_block(state: dict, checkpoint: str, count: int,
                 jump: dict, standing: dict) -> str:
    """Consume one evaluation, applying rollback and advancement decisions."""
    if count <= 0 or count > V5_BLOCK_UPDATES:
        raise ValueError("V5 block must contain 1..250 updates")
    if state["new_updates"] + count > V5_MAX_UPDATES:
        raise ValueError("V5 update budget exceeded")

    state["new_updates"] += count
    state["slice_updates"] += count
    phase = int(state["bootstrap_phase"])
    pending = state.get("pending_reward_change")
    rolled_back = bool(
        pending and v5_reward_regressed(
            jump, standing, pending["jump"], pending["standing"]
        )
    )
    row = {
        "baseline": False,
        "selected": not rolled_back,
        "candidate_checkpoint": checkpoint,
        "checkpoint": pending["checkpoint"] if rolled_back else checkpoint,
        "new_updates": state["new_updates"],
        "slice_updates": state["slice_updates"],
        "height_index": state["height_index"],
        "target_delta": target_delta(state),
        "required_delta": target_delta(state),
        "jump": jump,
        "standing": standing,
        "weights": weights(state),
        "reward_change": None if pending is None else pending["kind"],
    }
    state["evaluations"].append(row)

    if rolled_back:
        state["checkpoint"] = pending["checkpoint"]
        state.update(pending["weights"])
        if pending["kind"] == "strong_heading":
            state["heading_strengthen_blocked"] = True
        elif pending["kind"] == "height_2p0":
            state["height_weight_raise_blocked"] = True
        state["streak"] = 0
        state["no_improvement"] = 0
        state["status"] = "rollback_" + pending["kind"]
        state["pending_reward_change"] = None
    else:
        state["checkpoint"] = checkpoint
        accepted = {
            "height_index": state["height_index"],
            "jump": jump,
            "standing": standing,
            "checkpoint": checkpoint,
        }
        earlier = [
            r["jump"] for r in state["accepted_evaluations"]
            if r["height_index"] == state["height_index"]
        ]
        state["no_improvement"] = 0 if v5_improved(jump, earlier) else state["no_improvement"] + 1
        state["accepted_evaluations"].append(accepted)
        state["last_selected_evaluation"] = deepcopy(row)
        state["pending_reward_change"] = None

    # The first two fixed 125-update blocks, plus the optional heading block,
    # are repair setup. Their evaluations do not count toward the two-point
    # 3 cm consolidation requirement.
    if phase < 3:
        state["bootstrap_phase"] = phase + 1
        if state["bootstrap_phase"] >= 3:
            state["streak"] = 0
            state["no_improvement"] = 0
        return state["status"]

    if rolled_back:
        return state["status"]

    passed = v5_metrics_pass(jump, standing, target_delta(state))
    state["streak"] = state["streak"] + 1 if passed else 0
    required_streak = 3 if state["height_index"] == len(V5_HEIGHTS) - 1 else 2
    if state["streak"] >= required_streak:
        if state["height_index"] == len(V5_HEIGHTS) - 1:
            state["status"] = "candidate_passed"
            return state["status"]
        state["height_index"] += 1
        state["slice_updates"] = 0
        state["streak"] = 0
        state["no_improvement"] = 0
        state["status"] = "advance_height"
        return state["status"]

    cap = V5_HEIGHT_CAPS[int(state["height_index"])]
    if cap is not None and state["slice_updates"] >= cap:
        state["status"] = (
            "quality_diagnosis_heading"
            if state["height_index"] == 0 and not v5_heading_pass(jump)
            else "quality_diagnosis_slice_budget"
        )
        return state["status"]

    if state["no_improvement"] >= 3:
        if (state["height_weight"] == 1.5
                and not state["height_weight_raised"]
                and not state["height_weight_raise_blocked"]):
            _schedule_reward_change(state, "height_2p0", height_weight=2.0)
            state["height_weight_raised"] = True
            state["no_improvement"] = 0
            state["status"] = "increase_height_reward_2p0"
            return state["status"]
        state["status"] = "quality_diagnosis_no_improvement"
        return state["status"]

    if state["new_updates"] >= V5_MAX_UPDATES:
        state["status"] = "budget_exhausted"
    else:
        state["status"] = "continue"
    return state["status"]


def terminal(state: dict) -> bool:
    return state.get("status") in {
        "candidate_passed",
        "passed",
        "budget_exhausted",
        "quality_diagnosis_heading",
        "quality_diagnosis_slice_budget",
        "quality_diagnosis_no_improvement",
        "final_multiseed_failed",
        "cpu_transfer_failed",
        "video_generation_failed",
    }
