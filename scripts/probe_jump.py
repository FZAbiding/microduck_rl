#!/usr/bin/env python3
"""Bounded physical feasibility probes, never policy acceptance evidence."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from scipy.optimize import least_squares
from tensordict import TensorDict
from mjlab_microduck.jump_evaluation import make_eval_env, raw_sample, physical_trace
from mjlab_microduck.jump_transfer import import_actor
from mjlab_microduck.jump_control import JumpController
from mjlab_microduck.jump_artifacts import atomic_json


def run(candidates, output, perturb=False):
    n=len(candidates); env=make_eval_env(n,standing=True,seed=43 if perturb else 42)
    env.cfg.episode_length_s=10.
    teacher,_=import_actor('artifacts/official/alpha_stand.onnx');teacher.to(env.device)
    stand=json.loads(Path('artifacts/jump-v3/preflight/standing_gate.json').read_text())['stand_z']
    judge=JumpController(n,env.device,stand_z=stand)
    robot=env.scene['robot'];names=[n for n in robot.joint_names if not n.startswith('passive_')]
    model=env.sim.mj_model
    joint_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+name) for name in names]
    pitch=[i for i,name in enumerate(names) if any(k in name for k in ('hip_pitch','knee','ankle'))]
    addresses=model.jnt_qposadr[joint_ids]
    foot_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'robot/'+name) for name in ('left_foot_collision','right_foot_collision')]
    trace=[];qbase=None;delta=None;records=[]
    try:
        obs,_=env.reset()
        if perturb:
            q=robot.data.joint_pos.clone()+torch.randn_like(robot.data.joint_pos)*.005
            robot.write_joint_position_to_sim(q)
            env.sim.forward()
        for step in range(300):
            t=step*.02
            with torch.inference_mode():
                action=teacher(TensorDict(obs,[n]))
            if step==90:
                qbase=env.sim.data.qpos[0].cpu().numpy().copy()
                home=robot.data.default_joint_pos[0].cpu().numpy()
                data=mujoco.MjData(model); data.qpos[:]=qbase;mujoco.mj_forward(model,data)
                target=data.geom_xpos[foot_ids][:,[0,2]].copy()
                rotations=data.geom_xmat[foot_ids].reshape(2,3,3)
                angles=np.arctan2(rotations[:,0,2],rotations[:,2,2])
                cache={}
                for depth in sorted(set(c['depth'] for c in candidates)):
                    data.qpos[:]=qbase;data.qpos[2]-=depth
                    def residual(x):
                        data.qpos[addresses[pitch]]=x;mujoco.mj_forward(model,data)
                        r=data.geom_xmat[foot_ids].reshape(2,3,3)
                        a=np.arctan2(r[:,0,2],r[:,2,2])
                        return np.r_[(data.geom_xpos[foot_ids][:,[0,2]]-target).ravel(),.03*(a-angles)]
                    x0=qbase[addresses[pitch]]
                    solution=least_squares(residual,x0,bounds=(model.jnt_range[np.array(joint_ids)[pitch],0]+1e-5,
                        model.jnt_range[np.array(joint_ids)[pitch],1]-1e-5),max_nfev=200)
                    offset=np.zeros(14);offset[pitch]=solution.x-x0
                    cache[depth]=offset
                delta=torch.tensor(np.stack([cache[c['depth']] for c in candidates]),device=env.device,dtype=torch.float32)
                base_action=torch.tensor(qbase[addresses]-home,device=env.device,dtype=torch.float32)
                accepted=judge.press(torch.ones(n,device=env.device,dtype=torch.bool))
            if step>=90:
                elapsed=t-1.8
                factors=[]
                for c in candidates:
                    if elapsed<c['crouch_s']: factor=elapsed/c['crouch_s']
                    elif elapsed<c['crouch_s']+c['extend_s']:
                        factor=1-1.2*(elapsed-c['crouch_s'])/c['extend_s']
                    elif elapsed<c['crouch_s']+c['extend_s']+.10: factor=-.2
                    else: factor=0.
                    factors.append(factor)
                factors=torch.tensor(factors,device=env.device,dtype=torch.float32)
                absolute=torch.tensor([c['mode']=='target' for c in candidates],device=env.device)
                active=factors!=0
                action=torch.where((absolute&active)[:,None],base_action,action)+factors[:,None]*delta
            obs,_,terminated,timeout,_=env.step(action)
            s=raw_sample(env);force=env.scene.sensors['feet_ground_contact'].data.force.reshape(n,-1,3).norm(dim=-1).sum(-1)
            judge.update(step+1,**s,force=force)
            trace.append({k:v.detach().cpu().numpy().copy() for k,v in
                {**s,**physical_trace(env),'actions':action,'qpos':env.sim.data.qpos,'qvel':env.sim.data.qvel,
                 'force':force,'time':torch.full_like(s['z'],(step+1)*.02)}.items()})
            ids=(terminated|timeout).nonzero().flatten()
            if len(ids):
                # Stop failed slots in the ledger even though mjlab needs reset.
                obs,_=env.reset(env_ids=ids)
        for i,c in enumerate(candidates):
            records.append({**c,'probe_id':i,'takeoff':bool(judge.took_off[i]),
                'peak_delta':float(judge.peak[i]-stand) if judge.took_off[i] else 0.,
                'invalid':bool(judge.invalid[i]),'complete':bool(judge.complete[i]),
                'accepted':bool(judge.accepted[i])})
    finally:env.close()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'traces.npz',**{k:np.stack([r[k] for r in trace]) for k in trace[0]})
    atomic_json(output/'results.json',{'classification':'physical_feasibility_only','trajectories':n,'records':records})
    return records

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/jump-v3/probes'));p.add_argument('--retest-existing',action='store_true');a=p.parse_args()
    candidates=[dict(depth=float(d),crouch_s=c,extend_s=e,mode=m) for d,c,e,m in
        itertools.product(np.linspace(.005,.04,8),(.2,.35,.5,.7),(.02,.06,.12,.2),('target','feedback'))]
    assert len(candidates)+100<=1024
    records=json.loads((a.output/'sweep/results.json').read_text())['records'] if a.retest_existing else run(candidates,a.output/'sweep')
    best=sorted((r for r in records if r['takeoff']),key=lambda r:-r['peak_delta'])[:5]
    repeats=run([dict(r,repeat=i) for r in best for i in range(20)],a.output/'retest',True) if best else []
    atomic_json(a.output/'summary.json',{'trajectories':len(candidates)+len(repeats),
        'best':best,'retests':repeats,'classification':'physical_feasibility_only',
        'auxiliary_initial_states_verified':False,
        'note':'No state has been certified for reverse-curriculum initialization; probes are scripted forces via unchanged BAM targets.'})
