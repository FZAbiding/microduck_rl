"""Exact ONNX actor import; a transfer start has no optimizer/training history."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel


class FrozenNormalization(torch.nn.Module):
    def __init__(self, dim=61):
        super().__init__()
        self.register_buffer('mean', torch.zeros(1, dim))
        self.register_buffer('denominator', torch.ones(1, dim))

    def forward(self, x):
        return (x - self.mean) / self.denominator

    def update(self, x):
        pass


class TransferActor(MLPModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = FrozenNormalization(self.obs_dim)


    def configure_exploration(self, joint_names, branch='A'):
        if len(joint_names) != 14 or len(set(joint_names)) != 14:
            raise ValueError('Expected 14 named servo actions')
        device = next(self.parameters()).device
        head = [any(k in n for k in ('head_', 'neck_')) for n in joint_names]
        legs = [any(k in n for k in ('hip_pitch', 'knee', 'ankle')) for n in joint_names]
        idle = torch.tensor([.03 if h else .05 for h in head], device=device)
        fixed = torch.tensor([.20 if l else .03 if h else .05 for l,h in zip(legs,head)], device=device)
        if sum(legs) != 6 or sum(head) != 4:
            raise ValueError(f'Unexpected joint groups: {joint_names}')
        self.register_buffer('idle_std', idle)
        self.register_buffer('request_std_fixed', fixed)
        self.register_buffer('exploration_b', torch.tensor(branch == 'B', device=device))

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if not stochastic_output or not hasattr(self, 'idle_std'):
            return super().forward(obs, masks, hidden_state, stochastic_output)
        if masks is not None:
            from rsl_rl.utils import unpad_trajectories
            obs = unpad_trajectories(obs, masks)
        mean = super().forward(obs, hidden_state=hidden_state)
        # The very same stored, unnormalized observation is used during rollout
        # and PPO minibatch recomputation. All distribution properties, including
        # old/new log-prob, entropy and KL, read this conditional Normal.
        requested = obs['actor'][:, 48:49] > .5
        learned = self.distribution.std_param.clamp_min(1e-4)
        request_std = torch.where(self.exploration_b, self.request_std_fixed, learned)
        std = torch.where(requested, request_std, self.idle_std).expand_as(mean)
        self.distribution._distribution = torch.distributions.Normal(mean, std)
        return self.distribution.sample()


def import_actor(path: str | Path):
    """Accept only the verified affine-normalizer/ELU graph, fail closed otherwise."""
    import onnx
    from onnx import numpy_helper
    from tensordict import TensorDict

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    nodes = list(model.graph.node)
    if [n.op_type for n in nodes] != ['Sub', 'Div', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm', 'Elu', 'Gemm']:
        raise ValueError('Unsupported ONNX graph; expected affine normalization and 512/256/128 ELU actor')
    arrays = {v.name: numpy_helper.to_array(v).copy() for v in model.graph.initializer}
    previous = model.graph.input[0].name
    for n in nodes:
        if n.input[0] != previous:
            raise ValueError('Non-sequential ONNX graph')
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        expected = {'transB': 1, 'alpha': 1.0, 'beta': 1.0, 'transA': 0} if n.op_type == 'Gemm' else {'alpha': 1.0}
        if any(k not in expected or v != expected[k] for k, v in attrs.items()):
            raise ValueError(f'Unsupported attributes: {attrs}')
        if n.op_type == 'Gemm' and attrs.get('transB', 0) != 1:
            raise ValueError('Expected transposed Gemm weights')
        previous = n.output[0]
    if previous != model.graph.output[0].name:
        raise ValueError('Unexpected output')
    obs = TensorDict({'actor': torch.zeros(1, 61)}, [1])
    actor = TransferActor(obs, {'actor': ['actor']}, 'actor', 14,
                          hidden_dims=(512, 256, 128), activation='elu', obs_normalization=True,
                          distribution_cfg={'class_name': 'GaussianDistribution', 'init_std': .15, 'std_type': 'scalar'})
    with torch.no_grad():
        actor.obs_normalizer.mean.copy_(torch.from_numpy(arrays[nodes[0].input[1]]))
        actor.obs_normalizer.denominator.copy_(torch.from_numpy(arrays[nodes[1].input[1]]))
        if not (torch.isfinite(actor.obs_normalizer.denominator).all() and (actor.obs_normalizer.denominator > 0).all()):
            raise ValueError('Invalid normalization denominator')
        for i, n in zip((0, 2, 4, 6), (nodes[2], nodes[4], nodes[6], nodes[8])):
            actor.mlp[i].weight.copy_(torch.from_numpy(arrays[n.input[1]]))
            actor.mlp[i].bias.copy_(torch.from_numpy(arrays[n.input[2]]))
    return actor.eval(), {p.key: p.value for p in model.metadata_props}


def prepare_transfer(path: str | Path, output: str | Path):
    import onnxruntime as ort
    from tensordict import TensorDict
    teacher, metadata = import_actor(path)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    x = np.random.default_rng(42).normal(0, .5, (1024, 61)).astype(np.float32)
    # Official export has a fixed batch of one.
    expected = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0] for row in x])
    with torch.no_grad():
        actual = teacher(TensorDict({'actor': torch.from_numpy(x)}, [len(x)])).numpy()
    max_error = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, atol=5e-5, rtol=1e-5):
        raise ValueError(f'ONNX reconstruction differs: {max_error}')
    actor = copy.deepcopy(teacher)
    # Absorb the old zero-vx contribution into bias, then initialize an unscaled
    # request input with zero weight. Both request values initially mean stand.
    with torch.no_grad():
        norm = actor.obs_normalizer
        actor.mlp[0].bias.add_(actor.mlp[0].weight[:, 48] * (-norm.mean[0, 48] / norm.denominator[0, 48]))
        actor.mlp[0].weight[:, 48].zero_()
        norm.mean[0, 48] = 0
        norm.denominator[0, 48] = 1
        x[:, 48] = 0
        obs = TensorDict({'actor': torch.from_numpy(x)}, [len(x)])
        zero_error = float((actor(obs) - teacher(obs)).abs().max())
        if zero_error > 5e-5:
            raise ValueError(f'Zero-command preservation failed: {zero_error}')
    result = {'format': 'microduck-standing-init-v1', 'actor_state_dict': actor.state_dict(),
              'teacher_state_dict': teacher.state_dict(), 'source_metadata': metadata,
              'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              'onnx_max_error': max_error, 'zero_command_max_error': zero_error}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save(result, output)
    return {k: v for k, v in result.items() if not k.endswith('state_dict')}
