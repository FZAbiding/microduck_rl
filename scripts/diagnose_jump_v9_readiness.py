#!/usr/bin/env python3
"""No learning: measure pre-request readiness under inherited exploration noise."""
import argparse
from pathlib import Path
from dataclasses import asdict

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg
from mjlab_microduck.jump_artifacts import atomic_json, sha256
from mjlab_microduck.jump_control import JumpController
from mjlab_microduck.jump_evaluation import raw_sample
from mjlab_microduck.jump_runner import JumpRunner
from mjlab_microduck.jump_v9 import configure_v9, set_idle_exploration
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--episodes', type=int, default=100)
    a = p.parse_args()
    if Path(a.output).exists():
        p.error('refusing to overwrite diagnostic')
    cfg = configure_v9(make_microduck_jump_env_cfg(nominal_bootstrap=False, command_protocol=2))
    cfg.scene.num_envs, cfg.seed = a.episodes, 123
    command = cfg.commands['twist']
    command.standing_probability = 1.
    command.recovery_probability = command.single_probability = command.triple_probability = 0.
    env = ManagerBasedRlEnv(cfg, device=a.device)
    results = []
    try:
        vec = RslRlVecEnvWrapper(env)
        agent = load_rl_cfg('Mjlab-Jump-Flat-MicroDuck')
        agent.init_checkpoint = None
        runner = JumpRunner(vec, asdict(agent), device=a.device)
        runner.load(a.checkpoint, map_location=a.device)
        actor = runner.alg.actor.eval()
        internal = mdp.jump_controller(env)
        internal.policy_version = 9
        idle = actor.idle_std.clone()
        names = [n for n in env.scene['robot'].joint_names if not n.startswith('passive_')]
        for scale in (0., 1., .25, .10):
            torch.manual_seed(123)
            env.seed(123)
            obs, _ = vec.reset()
            strict = JumpController(a.episodes, a.device, .02, internal.stand_z, .032, .03, 9)
            warm = JumpController(a.episodes, a.device, .02, internal.stand_z, .032, .03, 9)
            warm.strict_first_request = False
            set_idle_exploration(actor, names, max(scale, 1e-6))
            longest = torch.zeros(a.episodes, device=a.device)
            terminations = 0
            for step in range(350):
                with torch.inference_mode():
                    actions = actor(obs, stochastic_output=scale > 0)
                obs, _, dones, _ = vec.step(actions)
                terminations += int(dones.sum())
                sample = raw_sample(env)
                metrics = mdp.jump_v9_measurements(env)
                for judge in (strict, warm):
                    judge.update(step+1, **sample, **metrics, force=torch.zeros_like(longest))
                longest = torch.maximum(longest, strict.ready_time)
            results.append(dict(idle_std_scale=scale, stochastic=scale > 0,
                strict_ready_at_7s=float((strict.ready_time >= 2.-1e-4).float().mean()),
                warm_first_ready_at_7s=float((warm.ready_time >= 2.-1e-4).float().mean()),
                strict_ever_ready=float((longest >= 2.-1e-4).float().mean()),
                max_strict_continuous_ready_s=float(longest.max()), terminations=terminations))
            print(results[-1], flush=True)
        actor.idle_std.copy_(idle)
    finally:
        env.close()
    atomic_json(a.output, dict(diagnostic_only=True, ppo_updates=0, checkpoint=a.checkpoint,
                checkpoint_sha256=sha256(a.checkpoint), episodes=a.episodes, seed=123, results=results))


if __name__ == '__main__':
    main()
