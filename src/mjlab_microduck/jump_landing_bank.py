"""Versioned physical landing states, without simulator/actuator history replay."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mjlab_microduck.jump_artifacts import sha256


def save_bank(path, rows, joint_names, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError('no safe landing states collected')
    values = {key: np.stack([r[key] for r in rows]) for key in rows[0]}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values, joint_names=np.asarray(joint_names),
        metadata=np.asarray(json.dumps(dict(schema=2 if 'action' in values else 1, **metadata))))
    return dict(path=str(path), sha256=sha256(path), samples=len(rows))


def load_bank(path, joint_names, nq, nv):
    """Fail closed on model mismatch, missing fields, and invalid physical states."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        if meta['schema'] not in (1, 2) or list(data['joint_names']) != list(joint_names):
            raise ValueError('landing bank model/joint contract mismatch')
        keys = ['qpos', 'qvel', 'joint_max', 'episode_id', 'offset_s']
        if meta['schema'] == 2:
            keys += ['action', 'prev_action', 'prev_prev_action', 'target_yaw', 'heading_error']
        result = {k: data[k].copy() for k in keys}
    n = len(result['qpos'])
    if n == 0 or result['qpos'].shape != (n, nq) or result['qvel'].shape != (n, nv):
        raise ValueError('landing bank state dimensions mismatch')
    if any(len(v) != n or not np.isfinite(v).all() for v in result.values()):
        raise ValueError('landing bank nonfinite/inconsistent arrays')
    if not np.allclose(np.linalg.norm(result['qpos'][:, 3:7], axis=1), 1., atol=1e-4):
        raise ValueError('landing bank invalid root quaternion')
    if meta['schema'] == 2 and any(result[k].shape != (n, 14)
            for k in ('action', 'prev_action', 'prev_prev_action')):
        raise ValueError('landing bank action dimensions mismatch')
    return result, meta
