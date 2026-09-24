"""Evidence gates for jump-v3. Updates are budgeted across all branches."""
from __future__ import annotations
from copy import deepcopy
import math

TARGETS = {0:.015, 1:.02, 2:.03, 3:.03, 4:.03, 5:.03}
DR = {0:0., 1:0., 2:0., 3:.25, 4:.5, 5:1.}
ACTION_WEIGHTS = (0., -.02, -.05, -.10)


def configure_stage(cfg, stage, dr_strength=None, action_weight=0.):
    if stage not in TARGETS or action_weight not in ACTION_WEIGHTS:
        raise ValueError('Unknown jump stage / action cost')
    cfg = deepcopy(cfg)
    strength = DR[stage] if dr_strength is None else dr_strength
    if not 0 <= strength <= 1:
        raise ValueError('DR strength outside [0,1]')
    cfg.curriculum.clear()
    cfg.rewards['action_rate_l2'].weight = action_weight
    if strength == 0:
        from mjlab_microduck.jump_evaluation import nominalize
        nominalize(cfg)
        return cfg
    def interval(value, center=0.):
        if isinstance(value, dict):
            return {k:interval(v,center) for k,v in value.items()}
        return tuple(center+strength*(x-center) for x in value)
    for name,event in cfg.events.items():
        params = event.params
        for key,value in list(params.items()):
            if key in ('ranges','scale_range','bias_range','alpha_range','kp_range','kd_range'):
                center = 1. if key in ('scale_range','kp_range','kd_range') or params.get('operation')=='scale' or name=='foot_friction' else 0.
                params[key] = interval(value, center)
            elif key in ('max_pitch_deg','max_roll_deg'):
                params[key] *= strength
        if name=='reset_base':
            for axis,value in params['pose_range'].items():
                if axis!='z':
                    params['pose_range'][axis] = interval(value)
        if name=='push_robot':
            amp = .08 if strength==1. else 0.
            params['velocity_range'] = {'x':(-amp,amp),'y':(-amp,amp)}
    for group in cfg.observations.values():
        for term in group.terms.values():
            term.delay_min_lag = int(term.delay_min_lag*strength+.5)
            term.delay_max_lag = int(term.delay_max_lag*strength+.5)
            for key in ('max_angle_deg',):
                if key in term.params:
                    term.params[key] *= strength
            if term.noise:
                for key in ('n_min','n_max','std'):
                    if hasattr(term.noise,key):
                        setattr(term.noise,key,getattr(term.noise,key)*strength)
    for actuator in cfg.scene.entities['robot'].articulation.actuators:
        actuator.delay_min_lag = int(actuator.delay_min_lag*strength+.5)
        actuator.delay_max_lag = int(actuator.delay_max_lag*strength+.5)
        actuator.vin_range = interval(actuator.vin_range,7.4)
        actuator.vin_drop_gain_range = interval(actuator.vin_drop_gain_range,.1)
    return cfg


def record_evaluation(state, iteration, nominal, dr=None, standing=None):
    history = state.setdefault('evaluations',[])
    if history and iteration <= history[-1]['iteration']:
        raise ValueError('Distinct increasing evaluated checkpoints required')
    stage = state['stage']
    stand = standing['success_rate'] if standing else nominal.get('standing_rate',0.)
    row = {'iteration':iteration,'stage':stage, 'nominal':nominal['success_rate'],
           'dr':(dr or {}).get('success_rate',0.), 'standing':stand,
           'height':nominal['height_rate'],'takeoff':nominal['takeoff_rate'],
           'peak':nominal['peak_delta_p90'], 'action_weight':state.get('action_weight',0.)}
    same = [r for r in history if r['stage']==stage]
    improved = not same or any(row[key] >= max(r[key] for r in same)+threshold-1e-9
                              for key,threshold in (('height',.05),('takeoff',.05),('peak',.002)))
    state['no_improvement'] = 0 if improved else state.get('no_improvement',0)+1
    history.append(row)
    # Smoothness rollback is its own boundary; no height / DR upgrade here.
    previous_weight = state.pop('smoothness_previous',None)
    if previous_weight is not None and row['takeoff'] < state.pop('smoothness_takeoff')-.05-1e-9:
        state['action_weight'] = previous_weight
        state['streak'] = 0
        state['smoothness_blocked'] = True
        return 'rollback_smoothness'
    passed = stand>=.95 and (row['height']>=.8 if stage<=1 else
                            row['nominal']>=.8 and (stage<3 or row['dr']>=.7))
    state['streak'] = state.get('streak',0)+1 if passed else 0
    required = 1 if stage in (1,3,4) else 3
    if state['streak'] >= required:
        if stage==5:
            return 'passed'
        state['stage'] += 1
        state['streak'] = 0
        state['no_improvement'] = 0
        return 'advance'
    # Alternate smoothness upgrades with physical difficulty changes, only
    # after small-jump consolidation and a successful current evaluation.
    weight = state.get('action_weight',0.)
    if stage>=1 and passed and weight!=ACTION_WEIGHTS[-1] and not state.get('smoothness_blocked'):
        state['smoothness_previous'] = weight
        state['smoothness_takeoff'] = row['takeoff']
        state['action_weight'] = ACTION_WEIGHTS[ACTION_WEIGHTS.index(weight)+1]
        state['streak'] = 0
        return 'increase_smoothness'
    if stage==0 and state['no_improvement']>=3:
        return 'diagnose'
    return 'continue'

# Jump-v4 continuation curriculum.  It deliberately lives beside the frozen
# v3 gate so old checkpoints and protocol-1 evaluators remain reproducible.
V4_HEADING_WEIGHT_STAGES = (-0.10, -0.20)
V4_YAW_RATE_WEIGHT_STAGES = (-0.01, -0.02)
V4_HEAD_BIAS_WEIGHT_STAGES = (0.0, 0.5, 1.0, 1.5)
V4_HEIGHTS = (0.030, 0.035, 0.040)
V4_MAX_UPDATES = 3000
V4_BLOCK_UPDATES = 250


def configure_v4(cfg, target_delta=0.030, heading_weight=-0.20,
                 yaw_rate_weight=-0.02, head_bias_weight=0.0):
    """Build one immutable v4 slice with protocol-2 observations and rewards."""
    from copy import deepcopy
    cfg = deepcopy(cfg)
    cfg.commands['twist'].protocol = 2
    cfg.rewards['action_rate_l2'].weight = -0.02
    cfg.rewards['jump_heading_error'].weight = heading_weight
    cfg.rewards['jump_yaw_rate'].weight = yaw_rate_weight
    cfg.rewards['jump_head_bias'].weight = head_bias_weight
    cfg.curriculum.clear()
    return cfg


def v4_posture_schedule(stage: int) -> tuple[float, float, float]:
    """Return (heading, yaw-rate, head-bias) for the five 125-update gates."""
    if stage < 0 or stage > 4:
        raise ValueError('v4 posture stage must be in [0, 4]')
    if stage == 0:
        return (-0.10, -0.01, 0.0)
    if stage == 1:
        return (-0.20, -0.02, 0.0)
    return (-0.20, -0.02, V4_HEAD_BIAS_WEIGHT_STAGES[stage - 1])


# Jump-v5 is a nominal height continuation from the final v4 checkpoint. The
# values here are consumed by the supervisor and CPU regression tests so the
# training and acceptance contracts cannot drift independently.
V5_HEIGHTS = (0.030, 0.035, 0.040, 0.045, 0.050)
V5_HEIGHT_CAPS = (750, 750, 1000, 1000, None)
V5_MAX_UPDATES = 5000
V5_BLOCK_UPDATES = 250
V5_HEAD_BIAS_WEIGHT = 0.5
V5_ACTION_WEIGHT = -0.02
V5_BASE_HEADING_WEIGHT = -0.20
V5_BASE_YAW_RATE_WEIGHT = -0.02
V5_STRONG_HEADING_WEIGHT = -0.30
V5_STRONG_YAW_RATE_WEIGHT = -0.03


def configure_v5(cfg, target_delta=0.030, height_weight=1.0,
                 heading_weight=V5_BASE_HEADING_WEIGHT,
                 yaw_rate_weight=V5_BASE_YAW_RATE_WEIGHT,
                 head_bias_weight=V5_HEAD_BIAS_WEIGHT):
    """Build an immutable nominal V5 height slice."""
    if target_delta not in V5_HEIGHTS:
        raise ValueError('unknown V5 height target')
    if height_weight not in (1.0, 1.5, 2.0):
        raise ValueError('unknown V5 height reward tier')
    if (heading_weight, yaw_rate_weight) not in (
        (V5_BASE_HEADING_WEIGHT, V5_BASE_YAW_RATE_WEIGHT),
        (V5_STRONG_HEADING_WEIGHT, V5_STRONG_YAW_RATE_WEIGHT),
    ):
        raise ValueError('unknown V5 heading reward tier')
    if head_bias_weight != V5_HEAD_BIAS_WEIGHT:
        raise ValueError('V5 head-bias weight is fixed at +0.5')
    cfg = deepcopy(cfg)
    from mjlab_microduck.jump_evaluation import nominalize
    nominalize(cfg)
    cfg.commands['twist'].protocol = 2
    cfg.rewards['action_rate_l2'].weight = V5_ACTION_WEIGHT
    cfg.rewards['jump_height_progress'].weight = height_weight
    cfg.rewards['jump_heading_error'].weight = heading_weight
    cfg.rewards['jump_yaw_rate'].weight = yaw_rate_weight
    cfg.rewards['jump_head_bias'].weight = head_bias_weight
    cfg.curriculum.clear()
    return cfg


def v5_heading_pass(jump):
    """Strict heading-only gate used before leaving the 3 cm repair slice."""
    return (jump.get('heading_change_p90_deg', float('inf')) <= 10.0
            and jump.get('heading_change_max_deg', float('inf')) <= 20.0
            and jump.get('max_heading_error_p90_deg', float('inf')) <= 10.0
            and jump.get('max_heading_error_max_deg', float('inf')) <= 20.0)


def v5_metrics_pass(jump, standing, target_delta):
    """Complete V5 boundary gate; head metrics are deliberately diagnostic."""
    return (jump.get('success_rate', 0.0) >= 0.95
            and jump.get('height_rate', 0.0) >= 0.95
            and jump.get('takeoff_rate', 0.0) >= 0.95
            and standing.get('standing_rate', 0.0) >= 0.95
            and jump.get('peak_delta_p10', 0.0) >= target_delta - 1e-9
            and v5_heading_pass(jump)
            and jump.get('drift_p90_m', float('inf')) <= 0.04)


def v5_improved(current, earlier):
    """Meaningful within-slice improvement, measured against all prior bests."""
    if not earlier:
        return True
    return any(
        current.get(key, 0.0) >= max(row.get(key, 0.0) for row in earlier) + threshold - 1e-9
        for key, threshold in (
            ('success_rate', 0.05), ('height_rate', 0.05), ('peak_delta_p10', 0.002)
        )
    )


def v5_reward_regressed(current_jump, current_standing, baseline_jump, baseline_standing):
    """Whether a just-introduced reward tier must be rolled back."""
    return (current_jump.get('takeoff_rate', 0.0) < baseline_jump.get('takeoff_rate', 0.0) - 0.05 - 1e-9
            or current_standing.get('standing_rate', 0.0) < baseline_standing.get('standing_rate', 0.0) - 0.05 - 1e-9
            or current_jump.get('heading_change_p90_deg', float('inf'))
            > baseline_jump.get('heading_change_p90_deg', float('inf')) + 5.0 + 1e-9
            or current_jump.get('max_heading_error_p90_deg', float('inf'))
            > baseline_jump.get('max_heading_error_p90_deg', float('inf')) + 5.0 + 1e-9)


# V6 preserves V5 and adds an explicit precision-vertical objective. Heights
# advance in 1 cm steps only after two strict fixed-sample passes.
V6_HEIGHTS = (0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100)
V6_HEIGHT_CAPS = (2000, 1250, 1250, 1500, 1500, 1750, 1750, None)
V6_MAX_UPDATES = 12000
V6_BLOCK_UPDATES = 250
V6_REWARD_WEIGHTS = {
    "jump_height_progress": 1.5,
    "jump_heading_error": -0.40,
    "jump_yaw_rate": -0.04,
    "jump_head_bias": 0.5,
    "action_rate_l2": -0.02,
    "jump_planar_displacement": -0.35,
    "jump_planar_velocity": -0.15,
    "jump_upright": -0.12,
    "jump_post_landing_pose": -0.10,
    "jump_launch_action_symmetry": -0.08,
}


def configure_v6(cfg, target_delta=0.030):
    """Build one nominal precision-vertical slice without mutating V5."""
    if not any(abs(target_delta - value) <= 1e-9 for value in V6_HEIGHTS):
        raise ValueError("unknown V6 height target")
    cfg = deepcopy(cfg)
    from mjlab.managers import RewardTermCfg
    from mjlab_microduck.jump_evaluation import nominalize
    from mjlab_microduck.tasks import mdp as microduck_mdp
    nominalize(cfg)
    cfg.commands["twist"].protocol = 2
    cfg.rewards["jump_height_progress"].weight = V6_REWARD_WEIGHTS["jump_height_progress"]
    cfg.rewards["jump_heading_error"].weight = V6_REWARD_WEIGHTS["jump_heading_error"]
    cfg.rewards["jump_yaw_rate"].weight = V6_REWARD_WEIGHTS["jump_yaw_rate"]
    cfg.rewards["jump_head_bias"].weight = V6_REWARD_WEIGHTS["jump_head_bias"]
    cfg.rewards["action_rate_l2"].weight = V6_REWARD_WEIGHTS["action_rate_l2"]
    for name, func, params in (
        ("jump_planar_displacement", microduck_mdp.jump_planar_displacement_cost, {"scale_m": 0.02}),
        ("jump_planar_velocity", microduck_mdp.jump_planar_velocity_cost, {"scale_m_s": 0.20}),
        ("jump_upright", microduck_mdp.jump_upright_cost, {"deadband_deg": 2.0, "cap_deg": 15.0}),
        ("jump_post_landing_pose", microduck_mdp.jump_post_landing_pose_cost, {"scale_rad": 0.20}),
        ("jump_launch_action_symmetry", microduck_mdp.jump_launch_action_symmetry_cost, {"scale_rad": 0.20}),
    ):
        cfg.rewards[name] = RewardTermCfg(
            func=func, weight=V6_REWARD_WEIGHTS[name], params=params
        )
    cfg.curriculum.clear()
    return cfg


def v6_metrics_pass(jump, standing, target_delta):
    """Strict nominal gate for visually vertical, state-restoring jumps."""
    return (jump.get("success_rate", 0.0) >= 0.95
            and jump.get("height_rate", 0.0) >= 0.95
            and jump.get("takeoff_rate", 0.0) >= 0.95
            and standing.get("standing_rate", 0.0) >= 0.95
            and jump.get("peak_delta_p10", 0.0) >= target_delta - 1e-9
            and jump.get("heading_change_p90_deg", float("inf")) <= 5.0
            and jump.get("heading_change_max_deg", float("inf")) <= 10.0
            and jump.get("max_heading_error_p90_deg", float("inf")) <= 5.0
            and jump.get("max_heading_error_max_deg", float("inf")) <= 10.0
            and jump.get("drift_p90_m", float("inf")) <= 0.020
            and jump.get("final_drift_p90_m", float("inf")) <= 0.010
            and jump.get("final_tilt_p90_deg", float("inf")) <= 3.0
            and jump.get("final_pose_l1_p90_rad", float("inf")) <= 0.10
            and jump.get("final_horizontal_speed_p90_m_s", float("inf")) <= 0.03)


# Jump-V7 holds height at 3 cm and trains landing recovery plus repeatability.
# It is nominal-only: no domain randomization is enabled in this foundation stage.
V7_TARGET_DELTA = 0.030
V7_MAX_UPDATES = 3000
V7_BLOCK_UPDATES = 250
V7_BRANCH_UPDATES = 500
V7_REWARD_WEIGHTS = {
    "jump_height_progress": 1.5,
    "jump_failure": -8.0,
    "jump_timeout": -3.0,
    "jump_complete": 3.0,
    "jump_landing_impact": -0.03,
    "action_rate_l2": -0.02,
    "jump_head_bias": 0.5,
    "jump_planar_displacement_launch": -0.15,
    "jump_planar_velocity_launch": -0.10,
    "jump_heading_error_launch": -0.25,
    "jump_yaw_rate_launch": -0.03,
    "jump_upright_launch": -0.08,
    "jump_planar_displacement_recovery": -0.03,
    "jump_heading_error_recovery": -0.05,
    "jump_planar_velocity_recovery": -0.25,
    "jump_yaw_rate_recovery": -0.08,
    "jump_upright_recovery": -0.40,
    "jump_post_landing_pose": -0.25,
    "jump_single_foot_support": -0.25,
    "jump_launch_action_symmetry": -0.08,
}


def configure_v7(cfg):
    """Build the fixed nominal 3 cm recovery/repeatability training slice."""
    from mjlab.managers import RewardTermCfg
    from mjlab_microduck.jump_evaluation import nominalize
    from mjlab_microduck.tasks import mdp as microduck_mdp

    cfg = deepcopy(cfg)
    nominalize(cfg)
    cfg.episode_length_s = 24.0
    command = cfg.commands["twist"]
    command.protocol = 2
    command.policy_version = 7
    command.standing_probability = 0.20
    command.triple_probability = 0.30
    command.request_times = (7.0, 14.0, 21.0)
    command.request_jitter_s = 0.5

    # Remove all earlier jump objectives so no 0.25 s success jackpot or
    # stale post-completion position/heading correction survives into V7.
    for name in tuple(cfg.rewards):
        if name.startswith("jump_"):
            del cfg.rewards[name]

    for name, quantity in (
        ("jump_height_progress", "height_progress"),
        ("jump_failure", "failure_event"),
        ("jump_timeout", "timeout_event"),
        ("jump_complete", "complete_event"),
        ("jump_landing_impact", "impact"),
    ):
        cfg.rewards[name] = RewardTermCfg(
            func=microduck_mdp.jump_v2_reward,
            weight=V7_REWARD_WEIGHTS[name],
            params={"quantity": quantity},
        )
    cfg.rewards["action_rate_l2"].weight = V7_REWARD_WEIGHTS["action_rate_l2"]
    cfg.rewards["jump_head_bias"] = RewardTermCfg(
        func=microduck_mdp.jump_head_bias_penalty,
        weight=V7_REWARD_WEIGHTS["jump_head_bias"],
        params={"tau_s": 1.0},
    )

    phased = (
        ("jump_planar_displacement_launch", microduck_mdp.jump_planar_displacement_cost,
         {"scale_m": 0.02, "phase": "launch_flight"}),
        ("jump_planar_velocity_launch", microduck_mdp.jump_planar_velocity_cost,
         {"scale_m_s": 0.20, "phase": "launch_flight"}),
        ("jump_heading_error_launch", microduck_mdp.jump_heading_error_cost,
         {"phase": "launch_flight"}),
        ("jump_yaw_rate_launch", microduck_mdp.jump_yaw_rate_cost,
         {"phase": "launch_flight"}),
        ("jump_upright_launch", microduck_mdp.jump_upright_cost,
         {"deadband_deg": 2.0, "cap_deg": 15.0, "phase": "launch_flight"}),
        ("jump_planar_displacement_recovery", microduck_mdp.jump_planar_displacement_cost,
         {"scale_m": 0.02, "phase": "post_contact"}),
        ("jump_planar_velocity_recovery", microduck_mdp.jump_planar_velocity_cost,
         {"scale_m_s": 0.20, "phase": "post_contact"}),
        ("jump_heading_error_recovery", microduck_mdp.jump_heading_error_cost,
         {"phase": "post_contact"}),
        ("jump_yaw_rate_recovery", microduck_mdp.jump_yaw_rate_cost,
         {"phase": "post_contact"}),
        ("jump_upright_recovery", microduck_mdp.jump_upright_cost,
         {"deadband_deg": 2.0, "cap_deg": 15.0, "phase": "post_contact"}),
        ("jump_post_landing_pose", microduck_mdp.jump_post_landing_pose_cost,
         {"scale_rad": 0.20, "phase": "post_contact"}),
        ("jump_single_foot_support", microduck_mdp.jump_single_foot_support_cost,
         {"phase": "post_contact"}),
        ("jump_launch_action_symmetry", microduck_mdp.jump_launch_action_symmetry_cost,
         {"scale_rad": 0.20, "phase": "launch_flight"}),
    )
    for name, func, params in phased:
        cfg.rewards[name] = RewardTermCfg(
            func=func, weight=V7_REWARD_WEIGHTS[name], params=params
        )
    cfg.curriculum.clear()
    return cfg


def v7_single_metrics_pass(metrics):
    """Fixed-seed 100 x 15 s single-jump acceptance contract."""
    return (
        metrics.get("success_rate", 0.0) >= 0.95
        and metrics.get("failure_rate", 1.0) == 0.0
        and metrics.get("body_contact_rate", 1.0) == 0.0
        and metrics.get("invalid_rate", 1.0) == 0.0
        and metrics.get("peak_delta_p10", 0.0) >= V7_TARGET_DELTA - 1e-9
        and metrics.get("final_drift_p90_m", float("inf")) <= 0.010
        and metrics.get("final_drift_max_m", float("inf")) <= 0.020
        and metrics.get("heading_change_p90_deg", float("inf")) <= 3.0
        and metrics.get("heading_change_max_deg", float("inf")) <= 5.0
        and metrics.get("recovery_tilt_p90_deg", float("inf")) <= 3.0
        and metrics.get("recovery_tilt_max_deg", float("inf")) <= 5.0
        and metrics.get("recovery_pose_l1_p90_rad", float("inf")) <= 0.08
        and metrics.get("final_horizontal_speed_max_m_s", float("inf")) <= 0.03
    )


def v7_triple_metrics_pass(metrics):
    """Fixed-seed 100 x 30 s three-request acceptance contract."""
    return (
        metrics.get("sequence_success_rate", 0.0) >= 0.95
        and metrics.get("jump_success_rate", 0.0) >= 0.98
        and metrics.get("failure_rate", 1.0) == 0.0
        and metrics.get("body_contact_rate", 1.0) == 0.0
        and metrics.get("invalid_rate", 1.0) == 0.0
        and metrics.get("peak_delta_p10", 0.0) >= V7_TARGET_DELTA - 1e-9
        and metrics.get("final_drift_p90_m", float("inf")) <= 0.010
        and metrics.get("final_drift_max_m", float("inf")) <= 0.020
        and metrics.get("heading_change_p90_deg", float("inf")) <= 3.0
        and metrics.get("heading_change_max_deg", float("inf")) <= 5.0
        and metrics.get("recovery_tilt_p90_deg", float("inf")) <= 3.0
        and metrics.get("recovery_tilt_max_deg", float("inf")) <= 5.0
        and metrics.get("recovery_pose_l1_p90_rad", float("inf")) <= 0.08
        and metrics.get("final_horizontal_speed_max_m_s", float("inf")) <= 0.03
    )


def v7_metrics_pass(single, triple):
    return v7_single_metrics_pass(single) and v7_triple_metrics_pass(triple)


def v7_branch_key(single, triple):
    """Lexicographic branch selection: safety, triple, single, then precision."""
    zero_falls = int(
        single.get("failure_rate", 1.0) == 0.0
        and triple.get("failure_rate", 1.0) == 0.0
        and single.get("body_contact_rate", 1.0) == 0.0
        and triple.get("body_contact_rate", 1.0) == 0.0
        and single.get("invalid_rate", 1.0) == 0.0
        and triple.get("invalid_rate", 1.0) == 0.0
    )
    return (
        zero_falls,
        float(triple.get("sequence_success_rate", 0.0)),
        float(single.get("success_rate", 0.0)),
        -float(max(single.get("final_drift_p90_m", float("inf")),
                   triple.get("final_drift_p90_m", float("inf")))),
        -float(max(single.get("heading_change_p90_deg", float("inf")),
                   triple.get("heading_change_p90_deg", float("inf")))),
    )

# Jump-V8: fixed 3.2 cm training potential with an exact 3.0 cm acceptance
# contract. The extra 2 mm is a shaping target only.
V8_TRAIN_TARGET_DELTA = 0.032
V8_REQUIRED_DELTA = 0.030
V8_MAX_UPDATES = 4000
V8_BLOCK_UPDATES = 250
V8_BRANCH_UPDATES = 500
V8_REWARD_WEIGHTS = {
    "jump_height_progress": 1.5, "jump_complete": 4.0,
    "jump_recovery_potential": 1.5, "jump_failure": -8.0,
    "jump_timeout": -3.0, "jump_recovery_timeout": -4.0,
    "jump_readiness_failure": -4.0, "jump_landing_impact": -0.03,
    "action_rate_l2": -0.02, "jump_head_bias": 0.5,
    "jump_planar_displacement_launch": -0.20,
    "jump_planar_velocity_launch": -0.10, "jump_heading_error_launch": -0.25,
    "jump_yaw_rate_launch": -0.03, "jump_upright_launch": -0.08,
    "jump_launch_action_symmetry": -0.10, "jump_touchdown_displacement": -0.50,
    "jump_settle_velocity": -0.25, "jump_settle_yaw_rate": -0.08,
    "jump_settle_upright": -0.40, "jump_settle_leg_home": -0.35,
    "jump_settle_foot_pose": -0.40, "jump_settle_single_foot": -0.30,
    "jump_settle_heading": -0.02,
}

def v8_distribution(consolidated=False):
    return ({"recovery": .10, "single": .20, "triple": .60, "standing": .10}
            if consolidated else
            {"recovery": .35, "single": .25, "triple": .30, "standing": .10})

def configure_v8(cfg, consolidated=False):
    from mjlab.managers import EventTermCfg, RewardTermCfg
    from mjlab_microduck.jump_evaluation import nominalize
    from mjlab_microduck.tasks import mdp as microduck_mdp
    cfg = deepcopy(cfg)
    nominalize(cfg)
    cfg.episode_length_s = 32.0
    command = cfg.commands["twist"]
    command.protocol = 2
    command.policy_version = 8
    command.target_delta = V8_TRAIN_TARGET_DELTA
    command.required_delta = V8_REQUIRED_DELTA
    dist = v8_distribution(consolidated)
    command.recovery_probability = dist["recovery"]
    command.standing_probability = dist["standing"]
    command.single_probability = dist["single"]
    command.triple_probability = dist["triple"]
    command.request_times = (7.0, 14.0, 21.0)
    command.request_jitter_s = .5
    cfg.events["jump_reset_state"] = EventTermCfg(
        func=microduck_mdp.jump_v8_reset, mode="reset")
    for name in tuple(cfg.rewards):
        if name.startswith("jump_"):
            del cfg.rewards[name]
    direct = (
        ("jump_height_progress", microduck_mdp.jump_v2_reward, {"quantity": "height_progress"}),
        ("jump_complete", microduck_mdp.jump_v2_reward, {"quantity": "complete_event"}),
        ("jump_recovery_potential", microduck_mdp.jump_recovery_potential, {}),
        ("jump_failure", microduck_mdp.jump_v2_reward, {"quantity": "failure_event"}),
        ("jump_timeout", microduck_mdp.jump_v2_reward, {"quantity": "timeout_event"}),
        ("jump_recovery_timeout", microduck_mdp.jump_v2_reward, {"quantity": "recovery_timeout_event"}),
        ("jump_readiness_failure", microduck_mdp.jump_v2_reward, {"quantity": "readiness_failure_event"}),
        ("jump_landing_impact", microduck_mdp.jump_v2_reward, {"quantity": "impact"}),
        ("jump_head_bias", microduck_mdp.jump_head_bias_penalty, {"tau_s": 1.0}),
        ("jump_touchdown_displacement", microduck_mdp.jump_touchdown_displacement_cost, {"scale_m": .02}),
    )
    for name, func, params in direct:
        cfg.rewards[name] = RewardTermCfg(func=func, weight=V8_REWARD_WEIGHTS[name], params=params)
    cfg.rewards["action_rate_l2"].weight = V8_REWARD_WEIGHTS["action_rate_l2"]
    phased = (
        ("jump_planar_displacement_launch", microduck_mdp.jump_planar_displacement_cost, {"scale_m": .02, "phase": "launch_flight"}),
        ("jump_planar_velocity_launch", microduck_mdp.jump_planar_velocity_cost, {"scale_m_s": .20, "phase": "launch_flight"}),
        ("jump_heading_error_launch", microduck_mdp.jump_heading_error_cost, {"phase": "launch_flight"}),
        ("jump_yaw_rate_launch", microduck_mdp.jump_yaw_rate_cost, {"phase": "launch_flight"}),
        ("jump_upright_launch", microduck_mdp.jump_upright_cost, {"deadband_deg": 2., "cap_deg": 15., "phase": "launch_flight"}),
        ("jump_launch_action_symmetry", microduck_mdp.jump_launch_action_symmetry_cost, {"scale_rad": .20, "phase": "launch_flight"}),
        ("jump_settle_velocity", microduck_mdp.jump_settle_velocity_cost, {}),
        ("jump_settle_yaw_rate", microduck_mdp.jump_settle_yaw_rate_cost, {}),
        ("jump_settle_upright", microduck_mdp.jump_settle_upright_cost, {}),
        ("jump_settle_leg_home", microduck_mdp.jump_robust_leg_home_cost, {"scale_rad": .10, "phase": "settle_hold"}),
        ("jump_settle_foot_pose", microduck_mdp.jump_foot_pose_cost, {"scale_rad": math.radians(3.), "phase": "settle_hold"}),
        ("jump_settle_single_foot", microduck_mdp.jump_single_foot_support_cost, {"phase": "settle_hold"}),
        ("jump_settle_heading", microduck_mdp.jump_settle_heading_cost, {}),
    )
    for name, func, params in phased:
        cfg.rewards[name] = RewardTermCfg(func=func, weight=V8_REWARD_WEIGHTS[name], params=params)
    cfg.curriculum.clear()
    return cfg

def configure_v8_consolidation(cfg):
    return configure_v8(cfg, consolidated=True)

def _v8_safe(m):
    return all(m.get(k, 1.) == 0. for k in (
        "failure_rate", "body_contact_rate", "invalid_rate", "timeout_rate",
        "recovery_timeout_rate", "readiness_failure_rate"))

def v8_intermediate_pass(single, triple):
    return (_v8_safe(single) and _v8_safe(triple)
            and single.get("request_acceptance_rate", 0.) >= 1.
            and triple.get("request_acceptance_rate", 0.) >= 1.
            and max(single.get("foot_pose_p90_deg", float("inf")),
                    triple.get("foot_pose_p90_deg", float("inf"))) <= 3.
            and max(single.get("recovery_tilt_p90_deg", float("inf")),
                    triple.get("recovery_tilt_p90_deg", float("inf"))) <= 4.
            and max(single.get("leg_home_max_p90_rad", float("inf")),
                    triple.get("leg_home_max_p90_rad", float("inf"))) <= .10
            and triple.get("jump_success_rate", 0.) >= .85
            and triple.get("sequence_success_rate", 0.) >= .70)

def v8_single_metrics_pass(m):
    return (_v8_safe(m) and m.get("success_rate", 0.) >= .95
            and m.get("peak_delta_p10", 0.) >= .03
            and m.get("final_drift_p90_m", float("inf")) <= .01
            and m.get("final_drift_max_m", float("inf")) <= .02
            and m.get("heading_change_p90_deg", float("inf")) <= 3.
            and m.get("heading_change_max_deg", float("inf")) <= 5.
            and m.get("recovery_tilt_p90_deg", float("inf")) <= 3.
            and m.get("recovery_tilt_max_deg", float("inf")) <= 5.
            and m.get("foot_pose_p90_deg", float("inf")) <= 2.
            and m.get("foot_pose_max_deg", float("inf")) <= 4.
            and m.get("ankle_error_p90_rad", float("inf")) <= .05
            and m.get("ankle_error_max_rad", float("inf")) <= .08
            and m.get("leg_home_max_p90_rad", float("inf")) <= .10
            and m.get("leg_home_max_rad", float("inf")) <= .15
            and m.get("final_horizontal_speed_max_m_s", float("inf")) <= .03
            and m.get("ready_latency_p90_s", float("inf")) <= 3.
            and m.get("ready_latency_max_s", float("inf")) <= 4.
            and m.get("complete_latency_p90_s", float("inf")) <= 6.
            and m.get("complete_latency_max_s", float("inf")) <= 7.
            and m.get("busy_stuck_rate", 1.) == 0.)

def v8_triple_metrics_pass(m):
    return (m.get("sequence_success_rate", 0.) >= .95
            and m.get("jump_success_rate", 0.) >= .98
            and v8_single_metrics_pass({**m, "success_rate": m.get("jump_success_rate", 0.)}))

def v8_metrics_pass(single, triple):
    return v8_single_metrics_pass(single) and v8_triple_metrics_pass(triple)

def v8_branch_key(single, triple):
    # Missing diagnostics are worst-case, but must remain JSON-finite so a failed
    # evaluation can still be recorded and drive the prescribed branch switch.
    worst = 1e9
    safety = int(_v8_safe(single) and _v8_safe(triple))
    acceptance = min(single.get("request_acceptance_rate", 0.),
                     triple.get("request_acceptance_rate", 0.))
    foot = max(single.get("foot_pose_p90_deg", worst),
               triple.get("foot_pose_p90_deg", worst))
    return (safety, acceptance, -foot, triple.get("sequence_success_rate", 0.),
            triple.get("jump_success_rate", 0.), single.get("success_rate", 0.),
            -max(single.get("final_drift_p90_m", worst),
                 triple.get("final_drift_p90_m", worst)),
            -max(single.get("recovery_tilt_p90_deg", worst),
                 triple.get("recovery_tilt_p90_deg", worst)))
