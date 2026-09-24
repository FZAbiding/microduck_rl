#!/usr/bin/env python3
"""Local serial supervisor: A/B exploration, evidence gates, 4500 updates / 8 h."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from mjlab_microduck.jump_artifacts import atomic_json, snapshot, sha256
from mjlab_microduck.jump_curriculum import record_evaluation, TARGETS, DR


class Supervisor:
    def __init__(self, args):
        self.args=args;self.root=args.output
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=open('/tmp/microduck_jump_gpu0.lock','a+')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        existing=json.loads((self.root/'status.json').read_text()) if (args.continue_existing and (self.root/'status.json').exists()) else None
        if (self.root/'status.json').exists() and not args.continue_existing:
            raise RuntimeError('Existing supervisor status: refusing to reset its budget. Use --continue-existing with an explicit checkpoint.')
        self.started=time.time() if existing is None else existing.get('started_at', time.time())
        self.deadline=time.time()+args.hours*3600 if existing is not None else self.started+args.hours*3600
        self.used=0 if existing is None else int(existing.get('executed_or_reserved_updates',0))
        self.saved=0 if existing is None else int(existing.get('saved_updates_across_attempts',0))
        self.attempt=0 if existing is None else len(existing.get('ledger',[]))
        self.checkpoint=None if existing is None else existing.get('checkpoint')
        self.branch='preflight';self.stage=0;self.last_eval=None;self.child=None
        self.ledger=[] if existing is None else existing.get('ledger',[]);self.candidates=[]
        self.snapshot=snapshot(self.root/('source_resume5' if existing is not None else 'source'))
        self.num_envs=args.num_envs
        self.status('starting')

    def status(self, status, **extra):
        atomic_json(self.root/'status.json',{'status':status,'pid':os.getpid(),
            'child_pid':self.child.pid if self.child else None,'heartbeat':time.time(),
            'started_at':self.started,'deadline':self.deadline,'branch':self.branch,'stage':self.stage,
            'executed_or_reserved_updates':self.used,'saved_updates_across_attempts':self.saved,
            'remaining_updates':4500-self.used,'checkpoint':self.checkpoint,
            'latest_evaluation':self.last_eval,'num_envs':self.num_envs,'ledger':self.ledger,**extra})

    def process(self, command, logfile, status, progress=None):
        if time.time()>=self.deadline: raise TimeoutError('8 hour wall-time limit')
        manifest=json.loads(Path(self.snapshot).read_text())
        changed=[name for name,digest in manifest['files'].items() if not Path(name).exists() or sha256(name)!=digest]
        if changed: raise RuntimeError(f'Code changed after snapshot; new configuration review required: {changed[:5]}')
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        env.update(CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
        with open(logfile,'w') as stream:
            self.child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            while self.child.poll() is None:
                details={}
                if progress and Path(progress).exists():
                    details['current_block']=json.loads(Path(progress).read_text())
                self.status(status,**details)
                if time.time()>=self.deadline:
                    os.killpg(self.child.pid,signal.SIGTERM)
                    try:self.child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.child.pid,signal.SIGKILL);self.child.wait()
                    break
                time.sleep(5)
            code=self.child.returncode
            self.child=None
        return code

    def block(self, branch, state, count, resume=None, smoke=False):
        self.branch=branch;self.stage=state['stage']
        start_used=self.used
        # A reservation survives supervisor failure. Once the child exits it is
        # replaced with measured consumption, or conservatively retained.
        if not smoke:
            if self.used+count>4500: raise RuntimeError('PPO budget exceeded')
            self.used+=count
        remaining=count;current=resume
        for retry in range(3):
            self.attempt+=1
            name=f'{self.attempt:03d}_{branch}_s{state["stage"]}'+('_smoke' if smoke else '')
            output=self.root/name
            state_path=self.root/f'{name}_curriculum.json';atomic_json(state_path,state)
            command=[sys.executable,'scripts/train_jump_transfer.py','--output',str(output),
                '--num-envs',str(64 if smoke else self.num_envs),'--max-iterations',str(remaining),
                '--stage',str(state['stage']),'--action-weight',str(state.get('action_weight',0.)),
                '--exploration',branch if branch in ('A','B') else state.get('branch','A'),
                '--snapshot',self.snapshot,'--curriculum-json',str(state_path)]
            command+=['--resume',str(current)] if current else ['--init-checkpoint',str(self.args.init_checkpoint)]
            code=self.process(command,self.root/f'{name}.log','smoke' if smoke else 'training',output/'progress.json')
            progress=json.loads((output/'progress.json').read_text()) if (output/'progress.json').exists() else None
            result=json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else None
            actual=progress['block_completed_updates'] if progress else 0
            if result:
                actual=result['block_completed_updates']
                self.saved+=0 if smoke else actual
                self.checkpoint=result['checkpoint']
                self.ledger.append({'attempt':name,'updates':actual,'kind':'smoke' if smoke else 'ppo',
                    'saved':actual,'lost':0,'checkpoint':self.checkpoint,'resumed_from':str(current)})
                atomic_json(self.root/'budget.json',self.ledger)
                return self.checkpoint
            # Nonzero exit with no update counter may have died inside an
            # update. Reserve entire attempted block as specified in the plan.
            consumed=min(remaining,actual+1) if progress else remaining
            self.ledger.append({'attempt':name,'updates':consumed,'kind':'smoke' if smoke else 'ppo',
                                'exit_code':code,'conservative':True,'lost':consumed})
            atomic_json(self.root/'budget.json',self.ledger)
            # Only process interruption is retried. Python exceptions require
            # diagnosis, never an automatic dependency/physics change.
            if time.time()>=self.deadline: raise TimeoutError('8 hour wall-time limit')
            if code>=0:
                if not smoke: self.used -= remaining-consumed
                raise RuntimeError(f'Worker failed ({code}); see {name}.log')
            import torch
            choices=[]
            for path in output.glob('*.pt'):
                data=torch.load(path,map_location='cpu',weights_only=False)
                choices.append((data['completed_updates'],path))
            if not choices or retry==2: raise RuntimeError('Interrupted worker cannot resume, or two retries exhausted')
            _,current=max(choices)
            # Lost updates also consume budget. Retry only the remaining part.
            remaining-=consumed
            if remaining<=0: raise RuntimeError('Interrupted block exhausted its update reservation')
        raise RuntimeError('Retry limit reached')

    def evaluate(self, checkpoint, mode='nominal', seed=123, standing=False, full=False):
        name=Path(checkpoint).parent.name+f'_{mode}_{seed}'+('_standing' if standing else '')
        output=self.root/'evaluations';output.mkdir(exist_ok=True)
        target=output/f'{name}.json'
        if not target.exists():
            command=[sys.executable,'scripts/evaluate_jump.py','--checkpoint-file',str(checkpoint),
                '--mode',mode,'--seed',str(seed),'--episodes','100','--json-out',str(target)]
            if standing: command+=['--standing']
            if full: command+=['--stage','5']
            if self.process(command,output/f'{name}.log','evaluating')!=0:
                raise RuntimeError(f'Evaluation failed: {target}')
        result=json.loads(target.read_text())
        if result['checkpoint_sha256']!=sha256(checkpoint): raise RuntimeError('Checkpoint identity mismatch')
        self.last_eval=str(target)
        return result

    def continuation(self, checkpoint, state):
        import torch
        path=Path(checkpoint).with_name(Path(checkpoint).stem+'_continuation.pt')
        if path.exists(): return str(path)
        data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        meta=data['infos']['jump_transfer']
        meta['curriculum']=state
        meta['run']['evaluated_stage']=meta['run']['next_stage']
        meta['run']['next_stage']=state['stage']
        # The evaluated weights / target / cfg stay untouched. Worker explicitly
        # applies the new stage and saves a new effective configuration.
        torch.save(data,path)
        return str(path)

    def deliver(self, checkpoint, status):
        if not checkpoint:return
        output=self.root/'delivery';output.mkdir(exist_ok=True)
        command=[sys.executable,'scripts/export.py','Mjlab-Jump-Flat-MicroDuck',
            '--checkpoint-file',str(checkpoint),'--onnx-file',str(output/'jump_candidate.onnx'),
            '--num-envs','1','--device','cuda:0']
        if self.process(command,output/'export.log','exporting')!=0: raise RuntimeError('Standard export failed')
        command=[sys.executable,'scripts/verify_jump_export.py','--checkpoint',str(checkpoint),
            '--onnx',str(output/'jump_candidate.onnx'),'--output',str(output/'onnx_check.json')]
        if self.process(command,output/'onnx_check.log','verifying_export')!=0:raise RuntimeError('ONNX parity failed')
        command=[sys.executable,'scripts/infer_policy.py','--jump',str(output/'jump_candidate.onnx'),
            '--headless','--seconds','30','--jump-at','10.5','16.5','22.5',
            '--video-out',str(output/'cpu_bam_uncut.mp4'),'--json-out',str(output/'cpu_bam.json')]
        code=self.process(command,output/'cpu_bam.log','cpu_rehearsal')
        replay=json.loads((output/'cpu_bam.json').read_text()) if code==0 else {}
        command=[sys.executable,'scripts/check_jump_video.py',str(output/'cpu_bam_uncut.mp4'),str(output/'video_check.json')]
        decoded=self.process(command,output/'video_check.log','checking_video')==0 if code==0 else False
        passed=status=='passed' and replay.get('successful_requests')==3 and replay.get('initial_standing_passed',False) and not replay.get('invalid',True) and decoded
        atomic_json(output/'manifest.json',{'status':'metrics_passed_video_review_pending' if passed else status if status!='passed' else 'cpu_transfer_failed',
            'accepted':False,'metrics_passed':passed,'visual_review_pending':True,'checkpoint':checkpoint,'checkpoint_sha256':sha256(checkpoint),
            'onnx_sha256':sha256(output/'jump_candidate.onnx'),'source':self.snapshot,
            'cpu_three_jumps':replay.get('successful_requests',0),'video_decoded':decoded,
            'last_checkpoint':self.checkpoint,'ledger':str(self.root/'budget.json')})
        (output/'结论与续训.md').write_text(f'''# 跳跃训练结果\n\n状态：{'数值与CPU请求指标通过，等待完整视频视觉检查' if passed else '跳跃未完成 / 尚未通过完整验收'}。\n\n训练状态：{status}。CPU BAM 三次请求完成数：{replay.get('successful_requests',0)} / 3。\n\n候选：`{checkpoint}`，SHA-256：`{sha256(checkpoint)}`。\n\n完整恢复使用 `uv run scripts/train_jump_transfer.py --resume {checkpoint} --output <新的目录> --max-iterations 250`，并传入 checkpoint 所记录的 `--stage` 和 `--action-weight`；所有后续更新必须继续计入总预算。仿真回合重新初始化，不能宣称逐位连续。\n\n模型未上传，未部署实机。不可用本候选或物理探针声称完成跳跃。原始评估与逐步轨迹位于 `../evaluations/`，预算见 `../budget.json`。\n''')

    def run(self):
        if self.args.continue_existing:
            import torch
            if not self.args.resume_checkpoint:
                raise RuntimeError('--resume-checkpoint is required with --continue-existing')
            state=json.loads((self.root/'curriculum.json').read_text())
            # Resume the exact saved curriculum slice; never infer stage from the
            # checkpoint filename or reset smoothness after an interruption.
            state['branch']='A'
            checkpoint=self.continuation(self.args.resume_checkpoint,state)
            self.branch='A'; self.stage=1
            while self.used < 4500:
                count=min(250,4500-self.used)
                checkpoint=self.block('A',state,count,checkpoint)
                try:
                    nominal=self.evaluate(checkpoint); standing=self.evaluate(checkpoint,standing=True)
                except Exception as error:
                    atomic_json(self.root/'resume_eval_failure.json',{'checkpoint':checkpoint,'error':str(error)})
                    self.status('infrastructure_failure',error=str(error)); raise
                dr=self.evaluate(checkpoint,'dr') if state['stage']>=3 else None
                status=record_evaluation(state,self.used,nominal,dr,standing)
                atomic_json(self.root/'curriculum.json',state); self.status(status)
                if status=='diagnose':
                    atomic_json(self.root/'diagnosis.json',{'status':'behavior_plateau','checkpoint':checkpoint,'nominal':nominal,'standing':standing,'history':state['evaluations']}); break
                if status in ('passed','multi_seed_failed'): break
                checkpoint=self.continuation(checkpoint,state)
            self.checkpoint=checkpoint; self.deliver(checkpoint,status if 'status' in locals() else 'budget_exhausted'); self.status(status if 'status' in locals() else 'budget_exhausted'); return
        # Prerequisites are generated and checked interactively before launch.
        for required in ('preflight/standing_gate.json','preflight/cpu_standing.json','preflight/verification.json','probes/summary.json'):
            if not (self.root/required).exists(): raise RuntimeError(f'Missing prerequisite: {required}')
        verification=json.loads((self.root/'preflight/verification.json').read_text())
        if not verification['passed']: raise RuntimeError('Preflight verification failed')
        self.num_envs=verification['training_num_envs']
        results={}
        for branch in ('A','B'):
            state={'stage':0,'streak':0,'evaluations':[],'no_improvement':0,'action_weight':0.,'branch':branch}
            checkpoint=self.block(branch,state,250)
            nominal=self.evaluate(checkpoint);standing=self.evaluate(checkpoint,standing=True)
            results[branch]={'checkpoint':checkpoint,'nominal':nominal,'standing':standing,'state':state}
        def score(branch):
            n=results[branch]['nominal']
            return (n['height_rate'],n['takeoff_rate'],n['peak_delta_p90'],-n['failure_rate'],branch=='A')
        selected=max(('A','B'),key=score)
        atomic_json(self.root/'comparison.json',{'selected':selected,'results':results})
        checkpoint=results[selected]['checkpoint'];state=results[selected]['state']
        # First evaluation establishes the plateau baseline and may count toward
        # the three distinct small-jump checkpoints.
        record_evaluation(state,250,results[selected]['nominal'],standing=results[selected]['standing'])
        checkpoint=self.continuation(checkpoint,state)
        status='budget_exhausted'
        while self.used<4500:
            stage=state['stage']
            ceiling=2000 if stage==0 else 3500 if stage<3 else 4500
            if self.used>=ceiling:break
            count=min(250,ceiling-self.used)
            checkpoint=self.block(selected,state,count,checkpoint)
            nominal=self.evaluate(checkpoint)
            standing=self.evaluate(checkpoint,standing=True)
            dr=self.evaluate(checkpoint,'dr') if stage>=3 else None
            iteration=self.used-250  # surviving branch plus all later updates
            status=record_evaluation(state,iteration,nominal,dr,standing)
            atomic_json(self.root/'curriculum.json',state)
            self.status(status)
            if stage==5 and nominal['success_rate']>=.8 and dr['success_rate']>=.7 and standing['success_rate']>=.95:
                self.candidates.append({'checkpoint':checkpoint,'nominal':nominal,'dr':dr})
            if status=='passed':
                accepted=None
                for row in sorted(self.candidates,key=lambda x:(-x['nominal']['success_rate'],-x['dr']['success_rate'],x['nominal']['landing_impact_mean'],x['checkpoint'])):
                    ok=True
                    for seed in (124,125):
                        n=self.evaluate(row['checkpoint'],seed=seed)
                        d=self.evaluate(row['checkpoint'],'dr',seed=seed,full=True)
                        s=self.evaluate(row['checkpoint'],seed=seed,standing=True)
                        ok &= n['success_rate']>=.8 and d['success_rate']>=.7 and s['success_rate']>=.95
                    if ok:accepted=row['checkpoint'];break
                if accepted:checkpoint=accepted;break
                status='multi_seed_failed';break
            if status=='diagnose':
                atomic_json(self.root/'diagnosis.json',{'status':'behavior_plateau','checkpoint':checkpoint,
                    'nominal':nominal,'standing':standing,'history':state['evaluations'],
                    'probes':str(self.root/'probes/summary.json'),
                    'reason':'Three evaluations without required improvement. No certified auxiliary initial state; stop for trace diagnosis.',
                    'checks':['Inspect request acceptance / timeouts','Inspect one-shot reward totals and signs',
                              'Compare pitch-chain joint motion / torque with probes','Inspect teacher_request_kl logs']})
                status='behavior_plateau';break
            checkpoint=self.continuation(checkpoint,state)
        if status not in ('passed','behavior_plateau','multi_seed_failed'):status='budget_exhausted'
        self.deliver(checkpoint,status)
        self.status('metrics_passed_video_review_pending' if status=='passed' else status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('artifacts/jump-v3'))
    p.add_argument('--init-checkpoint',type=Path,default=Path('artifacts/jump_v2_transfer/standing_init.pt'))
    p.add_argument('--num-envs',type=int,default=4096)
    p.add_argument('--hours',type=float,default=8.)
    p.add_argument('--continue-existing',action='store_true')
    p.add_argument('--resume-checkpoint',type=Path)
    a=p.parse_args();s=Supervisor(a)
    try:s.run()
    except Exception as error:
        (s.root/'failure.txt').write_text(traceback.format_exc())
        failure_status='time_budget_exhausted' if isinstance(error,TimeoutError) else 'infrastructure_failure'
        if s.checkpoint:
            try:
                s.deadline=max(s.deadline,time.time()+900)  # delivery only; no further PPO
                s.deliver(s.checkpoint,failure_status)
            except Exception:
                (s.root/'delivery_failure.txt').write_text(traceback.format_exc())
        s.status(failure_status,error=str(error))
        raise

if __name__=='__main__':main()
