"""CPU BAM jump demo; invoked by infer_policy.py --jump. One policy throughout."""
from __future__ import annotations
import contextlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import mujoco
from mjlab_microduck.jump_control import JumpController, command_protocol, yaw_from_quat, wrap_to_pi


def run(args):
    from infer_policy import load_bam_model, load_mujoco_with_bam, PolicyInference, TerminalInput
    if args.no_bam:
        raise ValueError('--jump rehearsal requires BAM')
    model, data, bam, _ = load_mujoco_with_bam(
        args.scene or 'src/mjlab_microduck/robot/microduck/scene.xml',
        load_bam_model(args.kp_fw, args.vin, args.current_limit), .002,
        args.vin_drop_gain, 6.)
    policy = PolicyInference(model, data, standing_onnx_path=args.jump,
        use_projected_gravity=True, new_cmd_obs=True, bam_ctrl=bam,
        delay_min_lag=0 if not args.delay else args.delay[0],
        delay_max_lag=0 if not args.delay else args.delay[-1])
    session = policy.ort_session
    if session.get_inputs()[0].shape[-1] != 61 or session.get_outputs()[0].shape[-1] != 14:
        raise ValueError('Jump policy must be 61 -> 14')
    metadata = session.get_modelmeta().custom_metadata_map
    protocol = command_protocol(metadata)
    stand_z = args.stand_z if args.stand_z is not None else float(metadata.get('jump_stand_z', .115))
    target_delta = float(metadata.get('jump_target_delta', .03))
    required_delta = (float(metadata['jump_required_delta'])
                      if 'jump_required_delta' in metadata else None)
    policy_version = int(metadata.get('jump_policy_version', protocol + 2))
    c = JumpController(
        1, dt=.02, stand_z=stand_z, target_delta=target_delta,
        required_delta=required_delta, policy_version=policy_version,
    )
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'trunk_base_freejoint')
    qa, va = model.jnt_qposadr[root], model.jnt_dofadr[root]
    data.qpos[qa:qa+7] = [0, 0, .125, 1, 0, 0, 0]
    data.qpos[policy.joint_qpos_indices] = policy.default_pose
    bam.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    soles = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in ('left_foot_collision', 'right_foot_collision')]
    if min(soles) < 0:
        raise ValueError('Missing foot geometry')
    # Match training contact settings on the actual CPU model.
    model.geom_condim[soles] = 3
    model.geom_priority[soles] = 1
    model.geom_friction[soles, 0] = 1.
    video_writer = renderer = None
    if args.video_out:
        import imageio.v2 as imageio
        Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(args.video_out, fps=50)
        renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .65, 135, -15
    camera.lookat[:] = [0, 0, .12]
    viewer_context = contextlib.nullcontext(None) if args.headless else mujoco.viewer.launch_passive(model, data)
    scheduled = list(sorted(args.jump_at))
    trace, presses = [], []
    height_window = []
    max_tilt = 0.
    body_contact = False
    def sample_controller(step):
        feet = np.zeros(2, dtype=bool)
        body, force = False, 0.
        for contact in data.contact:
            g1, g2 = contact.geom1, contact.geom2
            ground = g1 if model.geom_bodyid[g1] == 0 else g2 if model.geom_bodyid[g2] == 0 else None
            if ground is None:
                continue
            other = g2 if ground == g1 else g1
            if other in soles:
                feet[soles.index(other)] = True
            else:
                body = True
        for idx, contact in enumerate(data.contact):
            if contact.geom1 in soles or contact.geom2 in soles:
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, idx, f)
                force += np.linalg.norm(f[:3])
        z = float(data.qpos[qa+2])
        quat = data.qpos[qa+3:qa+7].copy()
        tilt = math.degrees(math.acos(np.clip(1 - 2 * np.square(quat[1:3]).sum(), -1, 1)))
        def tensor(x, dtype=torch.float32):
            return torch.tensor(np.asarray(x)[None], dtype=dtype)
        joint_l1 = float(np.mean(np.abs(
            data.qpos[policy.joint_qpos_indices] - policy.default_pose
        )))
        horizontal_speed = float(np.linalg.norm(data.qvel[va:va+2]))
        c.update(
            step, tensor(z), tensor(quat), tensor(data.qvel[va+2]),
            tensor(data.qvel[va+3:va+6]), tensor(feet, torch.bool),
            tensor(body, torch.bool), tensor(force), tensor(joint_l1),
            tensor(horizontal_speed),
        )
        return z, tilt, feet, body
    sample_controller(0)
    successful_requests = 0
    maneuver_results = []
    previous_complete = False
    try:
        with viewer_context as viewer, TerminalInput() as terminal:
            if viewer:
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = .65, 135, -15
                viewer.cam.lookat[:] = [0, 0, .12]
            step = 0
            print('J: request one jump; P: one manual push; Q: quit. Same policy holds idle and landing.')
            while (not args.headless or step * .02 < args.seconds) and (viewer is None or viewer.is_running()):
                started = time.monotonic()
                keys = terminal.get_keys()
                if 'q' in keys:
                    break
                pressed = 'j' in keys
                while scheduled and step * .02 >= scheduled[0]:
                    scheduled.pop(0)
                    pressed = True
                if 'p' in keys:
                    data.qvel[va] += .1
                if pressed:
                    quat_now = torch.tensor(data.qpos[qa+3:qa+7][None], dtype=torch.float32)
                    accepted = c.press(torch.ones(1, dtype=torch.bool), yaw_from_quat(quat_now)).item()
                    presses.append({'time': step*.02, 'accepted': accepted})
                    print(f'Jump request at {step*.02:.2f}s: {"accepted" if accepted else "ignored (not ready)"}')
                # Runtime keeps the shared 13D command block; jump-v4 owns
                # only twist=[request, 0, heading_error] and zero-pads the
                # legacy head/body command slots.
                policy.command[:] = 0.0
                policy.command[:3] = c.command(protocol).cpu().numpy()[0]
                action = policy.infer()
                if not np.isfinite(action).all():
                    raise ValueError('Nonfinite policy action')
                policy.apply_action(action)
                for _ in range(10):
                    bam.update()
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                z, tilt, feet, body = sample_controller(step+1)
                max_tilt = max(max_tilt, tilt)
                body_contact |= body
                if c.complete.item() and not previous_complete:
                    successful_requests += 1
                    maneuver_results.append({'complete_time':(step+1)*.02,'peak':c.peak.item()})
                previous_complete = c.complete.item()
                head_err = (data.qpos[policy.joint_qpos_indices[5:9]] - policy.default_pose[5:9]).tolist()
                trace.append({'time': (step+1)*.02, 'z': z, 'tilt_deg': tilt, 'feet': feet.tolist(),
                    'body_contact': body, 'request': c.request.item(), 'took_off': c.took_off.item(),
                    'peak': c.peak.item(), 'complete': c.complete.item(), 'impact': c.impact.item(),
                    'heading_error_rad': float(c.heading_error.item()), 'yaw_rate_rad_s': float(c.yaw_rate.item()),
                    'head_joint_error_rad': head_err, 'vz': float(data.qvel[va+2]), 'omega_xy': data.qvel[va+3:va+6].tolist()})
                # Calibrate the natural standing height before the first
                # scheduled request; accepted stays latched for the episode.
                if .5 <= (step+1)*.02 < 1.2 and not c.request.item() and feet.all() and tilt <= 15 and not body:
                    height_window.append(z)
                if renderer:
                    camera.lookat[:2] = data.qpos[qa:qa+2]
                    renderer.update_scene(data, camera=camera)
                    from PIL import Image, ImageDraw
                    frame=Image.fromarray(renderer.render())
                    draw=ImageDraw.Draw(frame)
                    draw.rectangle((0,0,640,48),fill='black')
                    draw.text((8,5),f't={(step+1)*.02:.2f}s  z={z:.4f}m  dz={z-stand_z:+.4f}m  vz={data.qvel[va+2]:+.3f}m/s',fill='white')
                    draw.text((8,25),f'request={int(c.request.item())}  flight={int(c.took_off.item())} landed={int(c.landed.item())} complete={int(c.complete.item())} tilt={tilt:.1f} yaw={math.degrees(c.heading_error.item()):+.1f}° head={math.degrees(np.mean(np.abs(head_err))):.1f}°',fill='white')
                    video_writer.append_data(np.asarray(frame))
                if viewer:
                    viewer.sync()
                    time.sleep(max(0, .02 - (time.monotonic() - started)))
                step += 1
    finally:
        if renderer:
            renderer.close()
        if video_writer:
            video_writer.close()
    result = {'seconds': len(trace)*.02, 'bam': True, 'onnx_sha256': __import__('hashlib').sha256(Path(args.jump).read_bytes()).hexdigest(), 'stand_z': stand_z, 'target_delta': target_delta, 'required_delta': required_delta, 'acceptance_delta': c.acceptance_delta, 'policy_version': policy_version, 'protocol': protocol, 'presses': presses,
        'max_tilt_deg': max_tilt, 'body_contact': body_contact,
        'invalid': c.invalid.item(), 'timeout': c.timeout.item(),
        'complete': c.complete.item(), 'natural_stand_z': float(np.median(height_window)) if height_window else None,
        'initial_standing_passed': all(r['tilt_deg']<=15 and not r['body_contact'] for r in trace if r['time']<=10.),
        'trace': trace, 'successful_requests': successful_requests, 'maneuvers': maneuver_results}
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'trace'}, indent=2))
    return result
