#!/usr/bin/env python3
"""Run one bounded V9 block; fail before learning if provenance is inconsistent."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import random
from pathlib import Path
import time

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg
from mjlab_microduck.jump_artifacts import atomic_json, sha256, snapshot
from mjlab_microduck.jump_runner import JumpRunner
from mjlab_microduck.jump_v9 import STAGES, configure_v9, set_learning_rate, set_idle_exploration
from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg
from mjlab_microduck.tasks.mdp import jump_controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--parent-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=tuple(STAGES), default='pilot')
    parser.add_argument('--max-iterations', type=int, default=50)
    parser.add_argument('--num-envs', type=int, default=4096)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--idle-std-scale', type=float, default=1.,
                        help='Absolute multiplier of canonical idle exploration; request noise unchanged')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--warm-first-request', action='store_true',
                        help='Explicit training-only relaxed first request; evaluator stays strict')
    a = parser.parse_args()
    if not 1 <= a.max_iterations <= 250:
        parser.error('block must contain 1..250 updates')
    if not np.isfinite(a.idle_std_scale) or not 0 < a.idle_std_scale <= 1:
        parser.error('idle std scale must be in (0, 1]')
    if sha256(a.resume) != a.parent_sha256:
        parser.error('parent SHA256 mismatch')
    if a.output.exists():
        parser.error('use a fresh attempt directory')
    a.output.mkdir(parents=True)
    source = snapshot(a.output/'source')
    cfg = configure_v9(make_microduck_jump_env_cfg(nominal_bootstrap=False, command_protocol=2),
                       a.stage, strict_first_request=not a.warm_first_request)
    cfg.seed = a.seed
    cfg.scene.num_envs = a.num_envs
    cfg.sim.nan_guard.enabled = True
    agent = load_rl_cfg('Mjlab-Jump-Flat-MicroDuck')
    agent.experiment_name = 'jump_v9'
    agent.run_name = a.output.name
    agent.init_checkpoint = None
    agent.resume = True
    agent.save_interval = 25
    agent.algorithm.learning_rate = a.learning_rate
    from copy import deepcopy
    from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG
    agent.algorithm.symmetry_cfg = deepcopy(SYMMETRY_CFG)
    atomic_json(a.output/'configuration.json', dict(
        arguments=vars(a), source=source, agent=asdict(agent), env=repr(asdict(cfg))))
    env = ManagerBasedRlEnv(cfg, device=a.device)
    try:
        vec = RslRlVecEnvWrapper(env)
        runner = JumpRunner(vec, asdict(agent), str(a.output), a.device)
        runner.load(str(a.resume), map_location=a.device)
        c = jump_controller(env)
        c.policy_version = 9
        c.target_delta, c.required_delta = .032, .030
        c.strict_first_request = not a.warm_first_request
        set_learning_rate(runner.alg, a.learning_rate)
        names = [n for n in env.scene['robot'].joint_names if not n.startswith('passive_')]
        idle_std = set_idle_exploration(runner.alg.actor, names, a.idle_std_scale)
        # Independent stage seeds must take effect after parent RNG restoration.
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        env.reset(seed=a.seed)
        baseline = runner.alg.completed_updates
        baseline_steps = env.common_step_counter
        runner.run_metadata = dict(format='microduck-jump-v9', policy_version=9,
            command_protocol=2, stage=a.stage, parent_checkpoint=str(a.resume),
            parent_sha256=a.parent_sha256, source=source,
            strict_first_request=c.strict_first_request, learning_rate=a.learning_rate,
            idle_std_scale=a.idle_std_scale, actual_idle_std=idle_std)
        terms = list(env.reward_manager.active_terms)
        masses = torch.zeros(len(terms), device=env.device)
        sample_steps = 0
        original_step = env.step

        def measured_step(actions):
            nonlocal sample_steps
            result = original_step(actions)
            weighted = env.reward_manager._step_reward
            if not torch.isfinite(weighted).all():
                raise RuntimeError('nonfinite reward terms')
            for i, name in enumerate(terms):
                if (env.reward_manager.get_term_cfg(name).weight < 0 or name == 'jump_head_bias'):
                    if (weighted[:, i] > 1e-6).any():
                        raise RuntimeError(f'positive penalty: {name}')
            masses.add_(weighted.mean(0) * env.step_dt)
            sample_steps += 1
            return result

        env.step = measured_step
        started = time.time()

        def progress(completed):
            delta = completed - baseline
            if env.common_step_counter - baseline_steps != delta * 24:
                raise RuntimeError('update/step mismatch')
            if any(g['lr'] != a.learning_rate for g in runner.alg.optimizer.param_groups):
                raise RuntimeError('learning rate changed unexpectedly')
            atomic_json(a.output/'progress.json', dict(
                completed_updates=completed, block_completed_updates=delta,
                elapsed_s=time.time()-started, actual_learning_rate=runner.alg.learning_rate,
                parent_sha256=a.parent_sha256, actor_dim=61, action_dim=14,
                penalty_sign_check='passed', idle_std_scale=a.idle_std_scale,
                actual_idle_std=runner.alg.actor.idle_std.detach().cpu().tolist(),
                reward_mass_per_env={n: float(masses[i]) for i, n in enumerate(terms)},
                measured_steps=sample_steps))

        runner.alg.progress_callback = progress
        obs = vec.get_observations()
        assert obs['actor'].shape[-1] == 61 and abs(env.step_dt - .02) < 1e-9
        runner.learn(a.max_iterations)
        checkpoint = a.output/f'checkpoint_{runner.alg.completed_updates:04d}.pt'
        runner.save(checkpoint)
        if a.export:
            runner.export_policy_to_onnx(str(a.output), 'policy.onnx')
        atomic_json(a.output/'result.json', dict(
            status='block_complete', checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
            parent_sha256=a.parent_sha256, stage=a.stage,
            actual_learning_rate=runner.alg.learning_rate, idle_std_scale=a.idle_std_scale,
            block_completed_updates=runner.alg.completed_updates-baseline,
            completed_updates=runner.alg.completed_updates))
    finally:
        env.close()


if __name__ == '__main__':
    main()
