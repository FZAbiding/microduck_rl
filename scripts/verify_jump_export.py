#!/usr/bin/env python3
"""Check standard ONNX export against the exact frozen-normalizer actor."""
import argparse
import numpy as np
import torch
import onnxruntime as ort
from tensordict import TensorDict
from mjlab_microduck.jump_transfer import TransferActor
from mjlab_microduck.jump_artifacts import atomic_json,sha256

p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--onnx',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
d=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
obs=TensorDict({'actor':torch.zeros(1,61)},[1])
actor=TransferActor(obs,{'actor':['actor']},'actor',14,hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
    distribution_cfg={'class_name':'GaussianDistribution','init_std':.15,'std_type':'scalar'})
state={k:v for k,v in d['actor_state_dict'].items() if k not in ('idle_std','request_std_fixed','exploration_b')}
actor.load_state_dict(state);actor.eval()
session=ort.InferenceSession(a.onnx,providers=['CPUExecutionProvider'])
assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
x=np.vstack([np.zeros((1,61),dtype=np.float32),np.random.default_rng(123).normal(0,.5,(128,61)).astype(np.float32)])
with torch.no_grad(): reference=actor(TensorDict({'actor':torch.from_numpy(x)},[len(x)])).numpy()
actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:r[None]})[0] for r in x])
error=float(np.max(np.abs(actual-reference)))
metadata=session.get_modelmeta().custom_metadata_map
checkpoint_format=d.get('format', 'unknown')
metadata_passed=True
expected={}
if checkpoint_format in ('microduck-jump-v5', 'microduck-jump-v6',
                         'microduck-jump-v7'):
    version=checkpoint_format.rsplit('v',1)[-1]
    contract=d['infos']['jump_transfer']
    expected={'jump_policy_version':version,'jump_command_protocol':'2',
              'jump_target_delta':str(contract['target_delta']),
              'jump_required_delta':str(contract['required_delta'])}
    if checkpoint_format == 'microduck-jump-v7':
        expected.update({
            'jump_ready_duration_s': '2.0',
            'jump_recovery_duration_s': '5.0',
            'jump_ready_tilt_deg': '3.0',
            'jump_ready_pose_l1_rad': '0.08',
            'jump_ready_horizontal_speed_m_s': '0.03',
            'jump_ready_yaw_rate_rad_s': '0.1',
            'jump_failure_tilt_deg': '45.0',
        })
    metadata_passed=(contract.get('required_delta') is not None
                     and abs(contract['required_delta']-contract['target_delta'])<=1e-9
                     and all(key in metadata for key in expected)
                     and metadata.get('jump_policy_version')==version
                     and metadata.get('jump_command_protocol')=='2'
                     and abs(float(metadata.get('jump_target_delta','nan'))-contract['target_delta'])<=1e-9
                     and abs(float(metadata.get('jump_required_delta','nan'))-contract['required_delta'])<=1e-9
                     and all(metadata.get(key) == value for key, value in expected.items()
                             if key not in ('jump_target_delta', 'jump_required_delta')))
passed=error<=1e-4 and metadata_passed
atomic_json(a.output,{'passed':passed,'max_absolute_error':error,'input_shape':[1,61],'output_shape':[1,14],
    'normalizer_baked':True,'checkpoint_format':checkpoint_format,
    'metadata_passed':metadata_passed,'expected_jump_metadata':expected,
    'actual_jump_metadata':{k:v for k,v in metadata.items() if k.startswith('jump_')},
    'checkpoint_sha256':sha256(a.checkpoint),'onnx_sha256':sha256(a.onnx)})
assert error<=1e-4,error
assert metadata_passed,metadata
