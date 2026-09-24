"""Jump transfer PPO with versioned, immutable full-state checkpoints."""
from __future__ import annotations
import copy
import json
import hashlib
import random
import os
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import torch
from mjlab.rl import RslRlOnPolicyRunnerCfg, MjlabOnPolicyRunner
from rsl_rl.algorithms import PPO


@dataclass
class JumpRunnerCfg(RslRlOnPolicyRunnerCfg):
    init_checkpoint: str | None = None
    exploration_branch: str = 'A'
    gate_directory: str = 'artifacts/jump-v3/preflight'


class TransferPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.completed_updates = 0
        self.progress_callback = None

    def update(self):
        observations = self.storage.observations.flatten(0, 1)
        examples = observations[observations['standing_mask'][:, 0].bool()].clone()
        requests = observations[observations['actor'][:, 48] > .5][:4096].clone()
        result = super().update()
        if examples.batch_size[0]:
            indices = torch.randperm(examples.batch_size[0], device=self.device)[:4096]
            examples = examples[indices]
            with torch.no_grad():
                target = self.teacher(examples)
                before = self.actor(requests).clone() if len(requests) else None
                if before is not None:
                    self.actor(requests, stochastic_output=True)
                    old_params = tuple(x.clone() for x in self.actor.output_distribution_params)
            imitation = (self.actor(examples) - target).square().mean()
            self.optimizer.zero_grad()
            (.1 * imitation).backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            result['standing_imitation_before'] = imitation.item()
            with torch.no_grad():
                result['standing_imitation_after'] = (self.actor(examples)-target).square().mean().item()
                if before is not None:
                    result['teacher_request_action_change'] = (self.actor(requests)-before).square().mean().item()
                    self.actor(requests, stochastic_output=True)
                    result['teacher_request_kl'] = self.actor.get_kl_divergence(old_params, self.actor.output_distribution_params).mean().item()
        self.completed_updates += 1
        if self.progress_callback:
            self.progress_callback(self.completed_updates)
        return result

    def save(self):
        return {**super().save(), 'teacher_state_dict': self.teacher.state_dict(),
                'learning_rate': self.learning_rate, 'completed_updates': self.completed_updates}

    def load(self, loaded_dict, load_cfg=None, strict=True):
        result = super().load(loaded_dict, load_cfg, strict)
        self.teacher.load_state_dict(loaded_dict['teacher_state_dict'])
        self.learning_rate = loaded_dict['learning_rate']
        self.completed_updates = loaded_dict['completed_updates']
        return result


class JumpRunner(MjlabOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        init = train_cfg.get('init_checkpoint')
        if init and train_cfg.get('resume'):
            raise ValueError('init-checkpoint and resume are mutually exclusive')
        super().__init__(env, train_cfg, log_dir, device)
        self.curriculum_state = {'stage': 0, 'streak': 0, 'evaluations': [], 'no_improvement': 0}
        self.transfer_metadata = None
        self.run_metadata = {}
        if init:
            data = torch.load(init, map_location=device, weights_only=False)
            if data.get('format') != 'microduck-standing-init-v1':
                raise ValueError('Unsupported standing transfer format')
            self.alg.actor.load_state_dict(data['actor_state_dict'])
            self.alg.teacher.load_state_dict(data['teacher_state_dict'])
            self.transfer_metadata = {k:v for k,v in data.items() if not k.endswith('state_dict')}
            gate_dir = Path(train_cfg.get('gate_directory', 'artifacts/jump-v3/preflight'))
            gate = json.loads((gate_dir/'standing_gate.json').read_text())
            cpu = json.loads((gate_dir/'cpu_standing.json').read_text())
            digest = hashlib.sha256(Path(init).read_bytes()).hexdigest()
            if not (gate['passed'] and gate['episodes'] >= 100 and gate['seconds'] >= 10
                    and gate['init_sha256'] == digest and cpu['seconds'] >= 10
                    and cpu['bam'] and not cpu['invalid'] and cpu['max_tilt_deg'] <= 15):
                raise ValueError('Standing transfer gates failed')
            self.transfer_metadata['standing_gate'] = gate
            from mjlab_microduck.tasks.mdp import jump_controller
            jump_controller(env.unwrapped).stand_z = gate['stand_z']
        names = [n for n in env.unwrapped.scene['robot'].joint_names if not n.startswith('passive_')]
        self.alg.actor.configure_exploration(names, train_cfg.get('exploration_branch','A'))

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if num_learning_iterations > 5 and not (self.transfer_metadata or {}).get('standing_gate'):
            raise ValueError('Requires verified standing transfer')
        self.current_learning_iteration = self.alg.completed_updates
        super().learn(num_learning_iterations, init_at_random_ep_len=False)
        self.current_learning_iteration = self.alg.completed_updates

    def save(self, path, infos=None):
        from mjlab_microduck.tasks.mdp import jump_controller
        c = jump_controller(self.env.unwrapped)
        protocol = int(self.run_metadata.get('command_protocol', getattr(self.env.unwrapped.command_manager.get_term('twist'), 'protocol', 1)))
        # Symmetry resolution injects the live env into the algorithm cfg.
        # Strip that non-serializable handle while preserving the executable cfg.
        agent_cfg = copy.copy(self.cfg)
        agent_cfg['algorithm'] = copy.copy(agent_cfg.get('algorithm', {}))
        symmetry_cfg = agent_cfg['algorithm'].get('symmetry_cfg')
        if isinstance(symmetry_cfg, dict):
            agent_cfg['algorithm']['symmetry_cfg'] = {
                key: value for key, value in symmetry_cfg.items() if key != '_env'}
        state = {'curriculum': copy.deepcopy(self.curriculum_state), 'transfer': self.transfer_metadata,
                 'stand_z': c.stand_z, 'target_delta': c.target_delta,
                 'required_delta': c.required_delta, 'command_protocol': protocol,
                 'policy_version': c.policy_version,
                 'recovery_contract': {
                     'ready_duration_s': c.ready_duration,
                     'completion_duration_s': c.completion_duration,
                     'ready_tilt_deg': 3.0 if c.policy_version >= 7 else 15.0,
                     'ready_pose_l1_rad': 0.08 if c.policy_version >= 7 else None,
                     'ready_horizontal_speed_m_s': 0.03 if c.policy_version >= 7 else None,
                     'ready_yaw_rate_rad_s': 0.1 if c.policy_version >= 7 else None,
                     'ready_joint_max_rad': 0.10 if c.policy_version >= 8 else (0.08 if c.policy_version >= 7 else None),
                     'ready_ankle_max_rad': 0.06 if c.policy_version >= 8 else None,
                     'ready_foot_pitch_deg': 3.0 if c.policy_version >= 8 else None,
                     'ready_foot_roll_deg': 3.0 if c.policy_version >= 8 else None,
                     'recovery_timeout_s': 8.0 if c.policy_version >= 8 else None,
                     'failure_tilt_deg': 45.0 if c.policy_version >= 7 else 70.0,
                 },
                 'posture_stage': int(self.run_metadata.get('posture_stage', 0)), 'run': self.run_metadata,
                 'reward_weights': {n:self.env.unwrapped.reward_manager.get_term_cfg(n).weight
                                    for n in self.env.unwrapped.reward_manager.active_terms},
                 'env_cfg': json.loads(json.dumps(asdict(self.env.unwrapped.cfg), default=lambda x: f'{x.__module__}:{x.__qualname__}' if callable(x) and hasattr(x, '__qualname__') else str(x))), 'agent_cfg': copy.deepcopy(agent_cfg)}
        checkpoint_format = self.run_metadata.get('format', 'microduck-jump-v3')
        if checkpoint_format in ('microduck-jump-v5', 'microduck-jump-v6',
                                  'microduck-jump-v7'):
            if (protocol != 2 or c.required_delta is None
                    or abs(c.required_delta - c.target_delta) > 1e-9):
                raise ValueError('V5+ checkpoint requires protocol 2 and an exact height gate')
        if checkpoint_format == 'microduck-jump-v7' and c.policy_version != 7:
            raise ValueError('V7 checkpoint requires the V7 recovery controller')
        if checkpoint_format == 'microduck-jump-v8' and (
                c.policy_version != 8 or abs(c.target_delta - .032) > 1e-9
                or c.required_delta is None or abs(c.required_delta - .030) > 1e-9):
            raise ValueError('V8 checkpoint requires the 3.2 cm / exact 3.0 cm contract')
        data = self.alg.save()
        data.update(format=checkpoint_format, iter=self.alg.completed_updates,
            infos={**(infos or {}), 'jump_transfer': state,
                   'env_state': {'common_step_counter':self.env.unwrapped.common_step_counter}},
            rng={'python': random.getstate(), 'numpy': np.random.get_state(),
                 'torch':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all()},
            simulation_resume='Episodes reinitialized; RNG and update counters restored; not bitwise continuous.')
        # Framework can request a duplicate final save at the same update.
        path = Path(path)
        if path.exists():
            old = torch.load(path, map_location='cpu', weights_only=False)
            if old.get('completed_updates') == self.alg.completed_updates:
                return
            raise FileExistsError(path)
        tmp = path.with_suffix('.tmp')
        torch.save(data, tmp)
        os.replace(tmp, path)

    def export_policy_to_onnx(self, path, filename='policy.onnx', verbose=False):
        super().export_policy_to_onnx(path, filename, verbose)
        import onnx
        from mjlab_microduck.tasks.mdp import jump_controller
        model = onnx.load(str(Path(path)/filename))
        metadata = {p.key:p.value for p in model.metadata_props}
        c = jump_controller(self.env.unwrapped)
        protocol = int(self.run_metadata.get('command_protocol', getattr(self.env.unwrapped.command_manager.get_term('twist'), 'protocol', 1)))
        raw_format_version = str(self.run_metadata.get('format', '')).rsplit('v', 1)[-1]
        policy_version = int(self.run_metadata.get(
            'policy_version', raw_format_version if raw_format_version.isdigit() else protocol + 2
        ))
        metadata.update(jump_stand_z=str(c.stand_z), jump_target_delta=str(c.target_delta),
                        jump_command_slot='48', jump_command_protocol=str(protocol),
                        jump_heading_error_scale='0.2', jump_heading_error_clip_rad='0.5',
                        jump_policy_version=str(policy_version),
                        jump_protocol=f'microduck-jump-v{policy_version}')
        if policy_version >= 7:
            metadata.update(
                jump_ready_duration_s=str(c.ready_duration),
                jump_recovery_duration_s=str(c.completion_duration),
                jump_ready_tilt_deg='3.0',
                jump_ready_pose_l1_rad='0.08',
                jump_ready_horizontal_speed_m_s='0.03',
                jump_ready_yaw_rate_rad_s='0.1',
                jump_failure_tilt_deg='45.0',
            )
        if c.required_delta is not None:
            metadata['jump_required_delta'] = str(c.required_delta)
        onnx.helper.set_model_props(model, metadata)
        onnx.save(model, str(Path(path)/filename))

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        data = torch.load(path, weights_only=False, map_location=map_location or self.device)
        if data.get('format') not in ('microduck-jump-v3', 'microduck-jump-v4',
                                      'microduck-jump-v5', 'microduck-jump-v6',
                                      'microduck-jump-v7', 'microduck-jump-v8'):
            raise ValueError('Unsupported jump checkpoint format')
        self.alg.load(data, load_cfg, strict)
        self.current_learning_iteration = data['completed_updates']
        state = data['infos']['jump_transfer']
        if data.get('format') in ('microduck-jump-v5', 'microduck-jump-v6',
                                   'microduck-jump-v7'):
            if (state.get('required_delta') is None
                    or abs(state['required_delta'] - state['target_delta']) > 1e-9
                    or int(state.get('command_protocol', 0)) != 2):
                raise ValueError('Invalid V5+ checkpoint controller contract')
        if data.get('format') == 'microduck-jump-v7':
            recovery = state.get('recovery_contract', {})
            if (int(state.get('policy_version', 0)) != 7
                    or float(recovery.get('ready_duration_s', 0.0)) != 2.0
                    or float(recovery.get('completion_duration_s', 0.0)) != 5.0):
                raise ValueError('Invalid V7 recovery controller contract')
        if data.get('format') == 'microduck-jump-v8':
            recovery = state.get('recovery_contract', {})
            if (int(state.get('policy_version', 0)) != 8
                    or abs(float(state.get('target_delta', 0.0)) - .032) > 1e-9
                    or abs(float(state.get('required_delta', 0.0)) - .030) > 1e-9
                    or float(recovery.get('ready_duration_s', 0.0)) != 2.0
                    or float(recovery.get('completion_duration_s', 0.0)) != 5.0
                    or float(recovery.get('recovery_timeout_s', 0.0)) != 8.0):
                raise ValueError('Invalid V8 recovery controller contract')
        self.curriculum_state = copy.deepcopy(state['curriculum'])
        self.transfer_metadata = state['transfer']
        self.run_metadata = copy.deepcopy(state['run'])
        from mjlab_microduck.tasks.mdp import jump_controller
        env = self.env.unwrapped
        env.common_step_counter = data['infos']['env_state']['common_step_counter']
        c = jump_controller(env)
        restore_jump_contract(c, state, env.command_manager.get_term('twist'))
        c.reset(torch.arange(env.num_envs, device=env.device), env.common_step_counter)
        random.setstate(data['rng']['python']); np.random.set_state(data['rng']['numpy'])
        torch.set_rng_state(data['rng']['torch'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in data['rng']['cuda']])
        return data['infos']


def restore_jump_contract(controller, state, command_term=None):
    """Restore the versioned controller contract without changing old gates."""
    controller.stand_z = state['stand_z']
    controller.target_delta = state['target_delta']
    controller.required_delta = state.get('required_delta')
    if 'policy_version' in state:
        controller.policy_version = int(state['policy_version'])
    if command_term is not None:
        command_term.protocol = int(state.get(
            'command_protocol', getattr(command_term, 'protocol', 1)
        ))
