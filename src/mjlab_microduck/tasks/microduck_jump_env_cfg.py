"""Microduck vertical jump task.

The policy learns one bounded, state-gated maneuver: crouch, push from both
feet, clear the floor, reach a measured peak above the standing equilibrium,
land on both feet, and remain upright.  It deliberately reuses the velocity
recipe so the deployed actor remains the shared 61D -> 14D policy interface.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_STANDUP_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    BODY_POSE_CMD_RESAMPLE_S,
    COM_RANDOMIZATION_RANGE,
    ENABLE_COM_RANDOMIZATION,
    ENABLE_HEAD_COM_RANDOMIZATION,
    HEAD_COM_RANDOMIZATION_RANGE,
    NUM_STEPS_PER_ENV,
    VELOCITY_PUSH_RANGE,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Measured natural HOME equilibrium. Keep this value tied to the settle check;
# it is a target in the task, not an arbitrary XML spawn height.
STAND_Z = 0.11581382155418396
JUMP_HEIGHT_DELTA = 0.030
TARGET_PEAK_Z = STAND_Z + JUMP_HEIGHT_DELTA
PEAK_TOLERANCE = 0.005
EPISODE_LENGTH_S = 6.0
STABLE_DURATION_S = 0.25
STAND_Z_TOLERANCE = 0.010
STABLE_TILT_DEG = 15.0
STABLE_MAX_VERTICAL_SPEED = 0.05
STABLE_MAX_HORIZONTAL_ANGULAR_SPEED = 0.5

# Training starts with a small target and low smoothness/impact costs.  The
# target and randomization are raised through explicit step curricula below.
STAGE0_TARGET_DELTA = 0.015
JUMP_ZERO_COMMAND_PROB = 0.25
HEAD_ZERO_COMMAND_PROB = 0.50
BODY_ZERO_COMMAND_PROB = 0.50


def _make_feet_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )


def _make_body_sensor() -> ContactSensorCfg:
    # All named collision geoms except the two soles. This is intentionally a
    # separate sensor: stable landing must reject a trunk/head ground contact.
    return ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(?!(ankle_left|ankle_right)$).*",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )


def _make_self_collision_sensor() -> ContactSensorCfg:
    return ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )


def make_microduck_jump_env_cfg(play: bool = False, nominal_bootstrap: bool = True, command_protocol: int = 1) -> ManagerBasedRlEnvCfg:
    """Create the flat Microduck jump configuration."""
    feet_sensor = _make_feet_sensor()
    body_sensor = _make_body_sensor()
    self_collision_sensor = _make_self_collision_sensor()

    cfg = make_microduck_velocity_env_cfg(play=play)
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors = (feet_sensor, body_sensor, self_collision_sensor)
    # make_velocity_env_cfg is the mjlab template, not the microduck wrapper:
    # remove its privileged-only height/base-velocity terms from the actor and
    # restore base velocity only in the critic.
    cfg.observations["actor"].terms.pop("base_lin_vel", None)
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0
    )
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.sim.nconmax = 80

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = 1.0

    # Remove gait and command-tracking terms. Keep only low dynamic blockers and
    # smoothness/safety costs; jump motion must remain affordable early.
    for name in (
        "track_linear_velocity",
        "track_angular_velocity",
        "upright",
        "pose",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "head_pose_tracking",
        "body_pose_tracking",
        "head_pose_bias",
        "soft_landing",
    ):
        cfg.rewards.pop(name, None)

    cfg.rewards["body_ang_vel"].params["asset_cfg"] = SceneEntityCfg(
        "robot", body_names=("trunk_base",)
    )
    cfg.rewards["body_ang_vel"].weight = -0.01
    cfg.rewards["angular_momentum"].weight = -0.005
    cfg.rewards["action_rate_l2"].weight = -0.10
    cfg.rewards["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_).*",)
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_sensor.name},
    )
    cfg.rewards["body_impact"] = RewardTermCfg(
        func=microduck_mdp.body_impact_cost,
        weight=-0.01,
        params={"sensor_name": body_sensor.name, "threshold": 1.0},
    )
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Reassert the BAM field expansion event explicitly: standalone configs
    # must carry it even when the base recipe changes.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields, mode="startup"
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history, mode="reset"
    )
    cfg.events["jump_reset_state"] = EventTermCfg(
        func=microduck_mdp.jump_v2_reset,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        "left_foot_collision",
        "right_foot_collision",
    )
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Start from the measured HOME equilibrium. Position noise is introduced by
    # curriculum only after the jump has been discovered.
    cfg.events["reset_base"].params["pose_range"]["z"] = (STAND_Z, STAND_Z)
    cfg.events["reset_robot_joints"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_).*",)
    )
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)

    # A push is useful for transfer, but is staged off in the bootstrap.
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0)
    }
    if play:
        cfg.events["push_robot"].params["velocity_range"] = {
            "x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE
        }

    # NaN guard includes the contact forces used by the privileged critic.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_sensor.name, body_sensor.name)},
    )

    cfg.commands['twist'] = microduck_mdp.JumpRequestCommandCfg(
        resampling_time_range=(1.e9, 1.e9), standing_probability=.25)
    cfg.commands['twist'].protocol = int(command_protocol)
    for name in ('head_pose', 'body_pose'):
        cfg.commands.pop(name, None)
    for group in ('actor', 'critic'):
        for name, dim in (('head_command', 4), ('body_command', 6)):
            cfg.observations[group].terms[name] = ObservationTermCfg(
                func=microduck_mdp.zero_command_padding, params={'dim': dim})

    # The base recipe already removes actor base_lin_vel, height scans, and
    # narrows joint selectors to the servo-only 14 joints. Make critic sensor
    # terms explicit and NaN safe for the contact-heavy jump.
    cfg.observations["critic"].terms.pop("foot_height", None)
    for name, safe_func in (
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
    ):
        if name in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[name].func = safe_func

    # Stages are changed only by the external checkpoint evaluation gate.
    cfg.curriculum.clear()
    for name in tuple(cfg.rewards):
        if name.startswith('jump_'):
            del cfg.rewards[name]
    for name, quantity, weight in (
        ('jump_height_progress', 'height_progress', 1.),
        ('jump_launch_velocity', 'launch_signal', 1.0),
        ('jump_launch_progress', 'launch_progress', 1.0),
        ('jump_landing_impact', 'impact', -.02),
        ('jump_success_score', 'success_event', 2.),
        ('jump_complete', 'complete_event', 1.),
        ('jump_failure', 'failure_event', -2.),
    ):
        cfg.rewards[name] = RewardTermCfg(func=microduck_mdp.jump_v2_reward,
            weight=weight, params={'quantity': quantity})
    # v3 keeps the legacy zero heading slot and no heading/head terms.  v4
    # explicitly opts into protocol 2 and starts from the fixed -0.02 action
    # smoothness weight used by the 2250 transfer checkpoint.
    cfg.rewards['action_rate_l2'].weight = -0.02 if command_protocol >= 2 else 0.
    cfg.rewards['jump_heading_error'] = RewardTermCfg(
        func=microduck_mdp.jump_heading_error_cost,
        weight=-0.10 if command_protocol >= 2 else 0.0)
    cfg.rewards['jump_yaw_rate'] = RewardTermCfg(
        func=microduck_mdp.jump_yaw_rate_cost,
        weight=-0.01 if command_protocol >= 2 else 0.0)
    cfg.rewards['jump_head_bias'] = RewardTermCfg(
        func=microduck_mdp.jump_head_bias_penalty, weight=0.0,
        params={'tau_s': 1.0})
    cfg.terminations['jump_failure'] = TerminationTermCfg(func=microduck_mdp.jump_v3_failure)
    cfg.observations['critic'].terms['jump_state'] = ObservationTermCfg(func=microduck_mdp.jump_v3_critic_state)
    cfg.events['jump_reset_state'] = EventTermCfg(func=microduck_mdp.jump_v2_reset, mode='reset')
    # Spawn above the floor; STAND_Z is measured under a closed-loop policy.
    cfg.events['reset_base'].params['pose_range']['z'] = (.125, .125)
    cfg.events['reset_base'].params['velocity_range'] = {k: (0., 0.) for k in
        ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
    cfg.events['reset_robot_joints'].params['position_range'] = (-.01, .01)
    cfg.events['push_robot'].params['velocity_range'] = {'x': (0., 0.), 'y': (0., 0.)}
    from mjlab.managers import ObservationGroupCfg
    cfg.observations['standing_mask'] = ObservationGroupCfg(terms={
        'standing_only': ObservationTermCfg(func=microduck_mdp.jump_v2_standing_mask)},
        enable_corruption=False)
    if nominal_bootstrap:
        from mjlab_microduck.jump_evaluation import nominalize
        nominalize(cfg)
    return cfg


from mjlab_microduck.jump_runner import JumpRunnerCfg


MicroduckJumpRlCfg = JumpRunnerCfg(
    actor=RslRlModelCfg(
        class_name="mjlab_microduck.jump_transfer:TransferActor",
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.15, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        class_name="mjlab_microduck.jump_runner:TransferPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    logger="tensorboard",
    upload_model=False,
    wandb_project="mjlab_microduck",
    experiment_name="jump_v3",
    run_name="jump_v3",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=4500,
)
