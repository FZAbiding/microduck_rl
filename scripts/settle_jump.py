#!/usr/bin/env python3
"""Headless HOME-pose settle check used before Jump training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microduck.tasks.microduck_jump_env_cfg import (
    STAND_Z,
    make_microduck_jump_env_cfg,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--joint-noise", type=float, default=0.03)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--standing-onnx", type=Path, default=Path("artifacts/official/alpha_stand.onnx"), help="Closed-loop standing actor")
    args = parser.parse_args()
    if args.seconds <= 0 or args.num_envs <= 0:
        raise ValueError("seconds and num-envs must be positive")

    cfg = make_microduck_jump_env_cfg(play=True)
    cfg.scene.num_envs = args.num_envs
    if not args.standing_onnx.exists():
        raise FileNotFoundError(args.standing_onnx)
    # This diagnostic must hold the robot with the verified standing actor.
    # Force every episode into the zero-request bucket; otherwise a scheduled
    # jump would make a settle check report a false failure.
    cfg.commands["twist"].standing_probability = 1.0
    cfg.auto_reset = False
    for name in (
        "push_robot",
        "randomize_com",
        "randomize_head_com",
        "randomize_mass_inertia",
        "randomize_joint_friction",
        "randomize_armature",
        "encoder_bias",
        "foot_friction",
        "base_com",
    ):
        cfg.events.pop(name, None)
    for name in ("com_range", "head_com_range", "jump_pushes"):
        cfg.curriculum.pop(name, None)
    cfg.events["reset_robot_joints"].params["position_range"] = (
        -args.joint_noise, args.joint_noise
    )
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
    # Keep the settle measurement about the pose/control equilibrium, not a
    # random world translation, yaw, or inherited base velocity.
    cfg.events["reset_base"].params["pose_range"].update(
        {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    )
    cfg.events["reset_base"].params["velocity_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    # Do not terminate the diagnostic at the first 70-degree tilt. A real
    # settle check must observe the actual posture for the full window and
    # report tilt/body contact, rather than turning a fallen state into a
    # seemingly clean reset.
    cfg.terminations.pop("fell_over", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    import onnxruntime as ort
    session = ort.InferenceSession(str(args.standing_onnx), providers=["CPUExecutionProvider"])
    obs, _ = env.reset()
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    max_tilt_deg = torch.zeros(env.num_envs, device=env.device)
    body_contact_seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    steps = int(round(args.seconds / env.step_dt))
    for _ in range(steps):
        obs, _, terminated, timed_out, _ = env.step(
            torch.as_tensor(__import__("numpy").concatenate([session.run(None, {session.get_inputs()[0].name: row[None]})[0] for row in obs["actor"].detach().cpu().numpy()]), device=env.device)
        )
        robot = env.scene["robot"]
        q = robot.data.root_link_quat_w
        cos_tilt = (1.0 - 2.0 * (q[:, 1].square() + q[:, 2].square())).clamp(-1.0, 1.0)
        tilt_deg = torch.rad2deg(torch.acos(cos_tilt))
        max_tilt_deg = torch.maximum(max_tilt_deg, tilt_deg)
        body_found = env.scene.sensors["body_ground_contact"].data.found
        body_contact_seen |= body_found.any(dim=tuple(range(1, body_found.ndim)))
        failed |= terminated | timed_out

    robot = env.scene["robot"]
    z = robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
    q = robot.data.root_link_quat_w
    cos_tilt = (1.0 - 2.0 * (q[:, 1].square() + q[:, 2].square())).clamp(-1.0, 1.0)
    tilt_deg = torch.rad2deg(torch.acos(cos_tilt))
    stable = (
        ~failed
        & (z - STAND_Z).abs().le(0.010)
        & tilt_deg.le(15.0)
        & max_tilt_deg.le(15.0)
        & ~body_contact_seen
        & robot.data.root_link_lin_vel_w[:, 2].abs().le(0.05)
    )
    result = {
        "seconds": args.seconds,
        "num_envs": args.num_envs,
        "joint_noise": args.joint_noise,
        "stand_z_config": STAND_Z,
        "stable_fraction": float(stable.float().mean().item()),
        "termination_fraction": float(failed.float().mean().item()),
        "trunk_z_median": float(z.median().item()),
        "trunk_z_p10": float(torch.quantile(z, 0.1).item()),
        "trunk_z_p90": float(torch.quantile(z, 0.9).item()),
        "tilt_deg_p90": float(torch.quantile(tilt_deg, 0.9).item()),
        "max_tilt_deg_p90": float(torch.quantile(max_tilt_deg, 0.9).item()),
        "body_contact_fraction": float(body_contact_seen.float().mean().item()),
        "passed": bool(stable.all().item()),
    }
    env.close()
    output = json.dumps(result, indent=2, sort_keys=True)
    print(output)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
