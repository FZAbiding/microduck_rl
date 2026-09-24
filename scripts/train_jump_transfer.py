#!/usr/bin/env python3
"""One immutable versioned jump PPO block; supervisors own global budgets."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import random
from pathlib import Path
import time
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg
from mjlab_microduck.jump_curriculum import (
    TARGETS, V6_HEIGHTS, V6_REWARD_WEIGHTS, V7_REWARD_WEIGHTS, V8_REWARD_WEIGHTS,
    configure_stage, configure_v4, configure_v5, configure_v6, configure_v7, configure_v8,
)
from mjlab_microduck.jump_runner import JumpRunner
from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg
from mjlab_microduck.tasks.mdp import jump_controller
from mjlab_microduck.jump_artifacts import atomic_json, snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group()
    source.add_argument('--init-checkpoint')
    source.add_argument('--resume',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=4096)
    p.add_argument('--max-iterations',type=int,default=250)
    p.add_argument('--stage',type=int,default=0)
    p.add_argument('--action-weight',type=float,default=0.)
    p.add_argument('--exploration',choices=['A','B'],default='A')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--snapshot')
    p.add_argument('--curriculum-json',type=Path)
    version = p.add_mutually_exclusive_group()
    version.add_argument('--v4', action='store_true')
    version.add_argument('--v5', action='store_true')
    version.add_argument('--v6', action='store_true')
    version.add_argument('--v7', action='store_true')
    version.add_argument('--v8', action='store_true')
    p.add_argument('--target-delta', type=float, default=0.03)
    p.add_argument('--required-delta', type=float, default=None)
    p.add_argument('--height-weight', type=float, default=1.0)
    p.add_argument('--heading-weight', type=float, default=-0.20)
    p.add_argument('--yaw-rate-weight', type=float, default=-0.02)
    p.add_argument('--head-bias-weight', type=float, default=0.0)
    p.add_argument('--posture-stage', type=int, default=0)
    a=p.parse_args()
    if not 1<=a.max_iterations<=250:
        p.error('One block must contain 1..250 updates')
    if a.v6 and not any(abs(a.target_delta - value) <= 1e-9 for value in V6_HEIGHTS):
        p.error('unknown V6 target height')
    if a.v7 and abs(a.target_delta - .03) > 1e-9:
        p.error('V7 target height is fixed at 3 cm')
    if a.v8 and (abs(a.target_delta - .032) > 1e-9
                 or a.required_delta is None or abs(a.required_delta - .030) > 1e-9):
        p.error('V8 requires --target-delta .032 and --required-delta .030')

    if a.v5:
        if a.target_delta not in (.030, .035, .040, .045, .050):
            p.error('unknown V5 target height')
        if a.height_weight not in (1.0, 1.5, 2.0):
            p.error('unknown V5 height reward tier')
        if (a.heading_weight, a.yaw_rate_weight) not in (
            (-.20, -.02), (-.30, -.03)
        ):
            p.error('unknown V5 heading reward tier')
        if a.head_bias_weight != .5:
            p.error('V5 head-bias weight is fixed at +0.5')
    if a.output.exists():
        p.error('Output directory already exists; each attempt requires a fresh path')
    a.output.mkdir(parents=True)
    snap=a.snapshot or snapshot(a.output/'source')
    if a.v8:
        cfg=configure_v8(make_microduck_jump_env_cfg(
            nominal_bootstrap=False, command_protocol=2))
    elif a.v7:
        if a.required_delta is None:
            p.error('--v7 requires --required-delta')
        if abs(a.required_delta - .03) > 1e-9:
            p.error('V7 requires exact 3 cm --required-delta')
        cfg=configure_v7(make_microduck_jump_env_cfg(
            nominal_bootstrap=False, command_protocol=2))
    elif a.v6:
        if a.required_delta is None:
            p.error('--v6 requires --required-delta')
        if abs(a.required_delta - a.target_delta) > 1e-9:
            p.error('V6 requires --required-delta == --target-delta')
        cfg=configure_v6(make_microduck_jump_env_cfg(
            nominal_bootstrap=False, command_protocol=2), a.target_delta)
    elif a.v5:
        if a.required_delta is None:
            p.error('--v5 requires --required-delta')
        if abs(a.required_delta - a.target_delta) > 1e-9:
            p.error('V5 requires --required-delta == --target-delta')
        cfg=configure_v5(make_microduck_jump_env_cfg(nominal_bootstrap=False, command_protocol=2),
                         target_delta=a.target_delta, height_weight=a.height_weight,
                         heading_weight=a.heading_weight, yaw_rate_weight=a.yaw_rate_weight,
                         head_bias_weight=a.head_bias_weight)
    elif a.v4:
        cfg=configure_v4(make_microduck_jump_env_cfg(nominal_bootstrap=False, command_protocol=2),
                         target_delta=a.target_delta, heading_weight=a.heading_weight,
                         yaw_rate_weight=a.yaw_rate_weight, head_bias_weight=a.head_bias_weight)
    else:
        cfg=configure_stage(make_microduck_jump_env_cfg(nominal_bootstrap=False),a.stage,action_weight=a.action_weight)
    cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed
    cfg.sim.nan_guard.enabled=True
    agent=load_rl_cfg('Mjlab-Jump-Flat-MicroDuck')
    if a.v4 or a.v5 or a.v6 or a.v7 or a.v8:
        # Keep continuation runs distinguishable while reusing the registered
        # runner/architecture for checkpoint compatibility.
        version_name = ('jump_v8' if a.v8 else 'jump_v7' if a.v7 else 'jump_v6' if a.v6
                        else 'jump_v5' if a.v5 else 'jump_v4')
        agent.experiment_name=version_name
        agent.run_name=version_name
        if a.v6 or a.v7 or a.v8:
            from copy import deepcopy
            from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG
            agent.algorithm.symmetry_cfg = deepcopy(SYMMETRY_CFG)
    agent.init_checkpoint=None if a.resume else a.init_checkpoint or 'artifacts/jump_v2_transfer/standing_init.pt'
    agent.resume=bool(a.resume);agent.seed=a.seed
    agent.exploration_branch=a.exploration
    agent.save_interval=25
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    atomic_json(a.output/'configuration.json',{'env':repr(asdict(cfg)),'agent':asdict(agent),
        'snapshot':snap,'seed':a.seed,'resume':str(a.resume),'episode_reinitialization':True})
    env=ManagerBasedRlEnv(cfg,device=a.device)
    try:
        vec=RslRlVecEnvWrapper(env)
        runner=JumpRunner(vec,asdict(agent),str(a.output),a.device)
        if a.resume:
            runner.load(str(a.resume),map_location=a.device)
            env.reset()
        if a.curriculum_json:
            runner.curriculum_state=json.loads(a.curriculum_json.read_text())
        runner.curriculum_state['stage']=a.stage
        runner.run_metadata={'snapshot':snap,'output':str(a.output),'exploration':a.exploration,
                             'evaluated_stage':a.stage,'next_stage':a.stage}
        if a.v4 or a.v5 or a.v6 or a.v7 or a.v8:
            version_number = 8 if a.v8 else 7 if a.v7 else 6 if a.v6 else 5 if a.v5 else 4
            runner.run_metadata.update(format=f'microduck-jump-v{version_number}',
                                       policy_version=version_number, command_protocol=2,
                                       posture_stage=a.posture_stage, target_delta=(.032 if a.v8 else a.target_delta),
                                       required_delta=(.030 if a.v8 else a.required_delta),
                                       height_weight=cfg.rewards['jump_height_progress'].weight,
                                       heading_weight=(None if a.v7 or a.v8 else cfg.rewards['jump_heading_error'].weight),
                                       yaw_rate_weight=(None if a.v7 or a.v8 else cfg.rewards['jump_yaw_rate'].weight),
                                       head_bias_weight=cfg.rewards['jump_head_bias'].weight,
                                       precision_reward_weights=(V8_REWARD_WEIGHTS if a.v8
                                                                 else V7_REWARD_WEIGHTS if a.v7
                                                                 else V6_REWARD_WEIGHTS if a.v6
                                                                 else None))
        c=jump_controller(env)
        c.target_delta=(.032 if a.v8 else a.target_delta if (a.v4 or a.v5 or a.v6 or a.v7) else TARGETS[a.stage])
        c.required_delta=(.030 if a.v8 else a.required_delta if (a.v5 or a.v6 or a.v7) else None)
        if a.v8:
            c.policy_version = 8
        elif a.v7:
            c.policy_version = 7
        baseline_updates=runner.alg.completed_updates
        baseline_steps=env.common_step_counter
        started=time.time()
        def progress(completed):
            steps=env.common_step_counter-baseline_steps
            delta=completed-baseline_updates
            if steps!=delta*24:
                raise RuntimeError(f'Update/step mismatch: {steps} vs {delta}*24')
            reward_terms=env.reward_manager.active_terms
            weighted=env.reward_manager._step_reward
            for idx,name in enumerate(reward_terms):
                weight = env.reward_manager.get_term_cfg(name).weight
                bad_sign = (weight < 0 and bool((weighted[:,idx]>1e-7).any())) or (name == 'jump_head_bias' and weight > 0 and bool((weighted[:,idx]>1e-7).any()))
                if bad_sign:
                    raise RuntimeError(f'Positive weighted penalty: {name}')
            atomic_json(a.output/'progress.json',{'pid':__import__('os').getpid(),'heartbeat':time.time(),
                'completed_updates':completed,'block_completed_updates':delta,
                'env_steps':steps,'samples':steps*a.num_envs,'elapsed_s':time.time()-started,
                'penalty_sign_check':'passed','actor_dim':61,'action_dim':14,
                'protocol':2 if (a.v4 or a.v5 or a.v6 or a.v7 or a.v8) else 1,
                'policy_version':8 if a.v8 else 7 if a.v7 else 6 if a.v6 else 5 if a.v5 else 4 if a.v4 else 3,
                'target_delta':c.target_delta,'required_delta':c.required_delta})
        runner.alg.progress_callback=progress
        obs=vec.get_observations()
        assert obs['actor'].shape[-1]==61 and runner.alg.actor.mlp[-1].out_features==14
        assert abs(env.step_dt-.02)<1e-8
        runner.learn(a.max_iterations)
        out=a.output/f'checkpoint_{runner.alg.completed_updates:04d}.pt'
        runner.save(out)
        # A separate sidecar keeps saved and executed updates explicit.
        atomic_json(a.output/'result.json',{'checkpoint':str(out),'completed_updates':runner.alg.completed_updates,
            'block_completed_updates':runner.alg.completed_updates-baseline_updates,
            'saved_updates':runner.alg.completed_updates,'env_steps':env.common_step_counter,
            'status':'block_complete'})
    finally:
        env.close()

if __name__=='__main__': main()
