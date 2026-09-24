"""50 Hz request protocol, shared verbatim by training and CPU BAM replay.

Events are unscaled request returns. MDP wrappers divide by dt exactly once.
Episode failures remain latched across requests; only reset clears them.
"""
from __future__ import annotations
import math
import torch

JUMP_COMMAND_PROTOCOL = 2
HEADING_ERROR_SCALE = 0.2
HEADING_ERROR_CLIP_RAD = 0.5


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap radians to [-pi, pi], including tensors on CUDA."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """World yaw from scalar-first quaternions."""
    w, x, y, z = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def encode_heading_error(error: torch.Tensor) -> torch.Tensor:
    """Protocol-2 raw observation encoding (bounded to +/-0.1)."""
    return HEADING_ERROR_SCALE * error.clamp(-HEADING_ERROR_CLIP_RAD, HEADING_ERROR_CLIP_RAD)


def command_protocol(metadata: dict[str, str] | None) -> int:
    """Missing v4 metadata means the legacy v3 protocol, by contract."""
    if not metadata:
        return 1
    try:
        return int(metadata.get("jump_command_protocol", "1"))
    except (TypeError, ValueError):
        return 1


class JumpController:
    def __init__(self, n, device='cpu', dt=.02, stand_z=.115, target_delta=.015,
                 required_delta=None, policy_version=3):
        self.n, self.device, self.dt = n, device, dt
        self.stand_z, self.target_delta = stand_z, target_delta
        self.policy_version = int(policy_version)
        # None is part of the backwards-compatibility contract: v3/v4
        # checkpoints predate this field and retain their historical 5 mm
        # acceptance margin. V5 always stores an explicit exact value.
        self.required_delta = required_delta
        self.state = {}
        for key in ('request', 'busy', 'accepted', 'took_off', 'landed', 'impact_paid',
                    'invalid', 'timeout', 'request_timeout', 'readiness_failure',
                    'success', 'complete', 'fresh', 'nan', 'fallen',
                    'recovery_timeout', 'settle_hold'):
            self.state[key] = torch.zeros(n, dtype=torch.bool, device=device)
        for key in ('ready_time', 'request_time', 'stable_time', 'peak', 'paid_height',
                    'prev_vz', 'paid_launch', 'height_progress', 'launch_signal',
                    'launch_progress', 'impact', 'episode_impact', 'success_event',
                    'complete_event', 'failure_event', 'timeout_event', 'target_yaw',
                    'current_yaw', 'heading_error', 'max_heading_error', 'yaw_rate',
                    'joint_l1', 'horizontal_speed', 'tilt'):
            self.state[key] = torch.zeros(n, device=device)
        for key in ('joint_max', 'ankle_max', 'foot_pitch_max', 'foot_roll_max',
                    'recovery_time'):
            self.state[key] = torch.zeros(n, device=device)
        for key in ('readiness_failure_event', 'recovery_timeout_event'):
            self.state[key] = torch.zeros(n, device=device)
        self.state['target_xy'] = torch.zeros(n, 2, device=device)
        self.state['air_steps'] = torch.zeros(n, dtype=torch.long, device=device)
        self.last_step = torch.full((n,), -1, dtype=torch.long, device=device)
        self.reset(torch.arange(n, device=device))

    @property
    def acceptance_delta(self):
        if self.required_delta is not None:
            return float(self.required_delta)
        # v4 keeps a 5 mm acceptance margin: 3.5 cm target accepts 3.0 cm
        # and 4.0 cm target accepts 3.5 cm; the legacy 3 cm slice accepts 2.5 cm.
        return max(self.target_delta - .005, 0.0) if self.target_delta >= .03 - 1e-6 else self.target_delta

    @property
    def ready_duration(self):
        return 2.0 if self.policy_version >= 7 else 1.0

    @property
    def completion_duration(self):
        return 5.0 if self.policy_version >= 7 else 1.25

    def phase_mask(self, phase: str) -> torch.Tensor:
        """Return per-request phase masks used by phase-weighted rewards."""
        valid = self.busy & ~self.invalid
        masks = {
            'launch': valid & self.request & ~self.took_off,
            'flight': valid & self.took_off & ~self.impact_paid,
            'touchdown': valid & self.impact_paid & ~self.landed,
            'recovery': valid & self.landed & ~self.complete,
            'launch_flight': valid & ~self.impact_paid,
            'post_contact': valid & self.impact_paid & ~self.complete,
            'settle_hold': self.settle_hold & ~self.invalid,
            'idle': ~self.busy,
            # Compatibility default for V3--V6 rewards.
            'accepted': self.accepted & ~self.invalid,
        }
        if phase not in masks:
            raise ValueError(f'unknown jump phase: {phase}')
        return masks[phase]

    def __getattr__(self, name):
        state = self.__dict__.get('state', {})
        if name in state:
            return state[name]
        raise AttributeError(name)

    def reset(self, ids, step=-1):
        for value in self.state.values():
            value[ids] = 0
        self.fresh[ids] = True
        self.last_step[ids] = step

    def press(self, pressed, current_yaw=None, current_xy=None):
        accepted = (pressed & ~self.busy
                    & (self.ready_time >= self.ready_duration - 1e-4)
                    & ~self.invalid)
        for key in ('took_off', 'landed', 'impact_paid', 'success', 'complete', 'request_timeout', 'recovery_timeout'):
            self.state[key][accepted] = False
        for key in ('stable_time', 'request_time', 'peak', 'air_steps', 'paid_launch', 'paid_height', 'recovery_time'):
            self.state[key][accepted] = 0
        self.request[accepted] = True
        self.busy[accepted] = True
        self.accepted |= accepted
        self.ready_time[accepted] = 0
        self.settle_hold[accepted] = False
        if current_yaw is None:
            current_yaw = self.current_yaw
        self.target_yaw[accepted] = wrap_to_pi(current_yaw[accepted])
        if current_xy is not None:
            self.target_xy[accepted] = current_xy[accepted]
        self.heading_error[accepted] = 0.0
        self.max_heading_error[accepted] = 0.0
        return accepted

    def mark_readiness_failure(self, rejected: torch.Tensor) -> None:
        """Record a rejected scheduled request without poisoning the episode."""
        rejected = rejected.to(dtype=torch.bool, device=self.device)
        fresh = rejected
        self.readiness_failure |= rejected
        self.readiness_failure_event[fresh] = 1.0

    def command(self, protocol=JUMP_COMMAND_PROTOCOL):
        """Return the shared twist=[request, 0, heading-error] command."""
        value = torch.zeros(self.n, 3, device=self.device)
        value[:, 0] = self.request.float()
        if protocol >= 2:
            value[:, 2] = encode_heading_error(self.heading_error)
        return value

    def update(self, step, z, quat, vz, omega_xy, feet, body, force,
               joint_l1=None, horizontal_speed=None, joint_max=None,
               ankle_max=None, foot_pitch_max=None, foot_roll_max=None):
        active = self.last_step != step
        if not active.any():
            return
        if joint_l1 is None:
            joint_l1 = torch.zeros_like(z)
        if horizontal_speed is None:
            horizontal_speed = torch.zeros_like(z)
        def _scalar(value):
            if value is None:
                return torch.zeros_like(z)
            value = value.to(device=z.device, dtype=z.dtype)
            return value if value.ndim == 1 else value.abs().amax(dim=-1)
        joint_max = _scalar(joint_max)
        ankle_max = _scalar(ankle_max)
        foot_pitch_max = _scalar(foot_pitch_max)
        foot_roll_max = _scalar(foot_roll_max)
        finite = (torch.isfinite(z) & torch.isfinite(quat).all(-1)
                  & torch.isfinite(vz) & torch.isfinite(omega_xy).all(-1)
                  & torch.isfinite(force) & torch.isfinite(joint_l1)
                  & torch.isfinite(horizontal_speed)
                  & torch.isfinite(joint_max) & torch.isfinite(ankle_max)
                  & torch.isfinite(foot_pitch_max) & torch.isfinite(foot_roll_max))
        yaw_rate = omega_xy[:, 2] if omega_xy.shape[-1] >= 3 else torch.zeros_like(vz)
        omega_xy = omega_xy[:, :2]
        current_yaw = yaw_from_quat(quat)
        cos_tilt = 1 - 2 * (quat[:, 1].square() + quat[:, 2].square())
        tilt_limit_deg = 45.0 if self.policy_version >= 7 else 70.0
        bad = ~finite | body | (cos_tilt < math.cos(math.radians(tilt_limit_deg)))
        for key in ('height_progress', 'launch_signal', 'launch_progress', 'impact',
                    'success_event', 'complete_event', 'failure_event', 'timeout_event',
                    'readiness_failure_event', 'recovery_timeout_event'):
            self.state[key][active] = 0
        self.failure_event[active & bad & ~self.invalid] = 1.
        self.nan |= active & ~finite
        self.fallen |= active & (body | (cos_tilt < math.cos(math.radians(tilt_limit_deg))))
        self.invalid |= active & bad
        self.current_yaw[active] = torch.nan_to_num(current_yaw[active])
        self.yaw_rate[active] = torch.nan_to_num(yaw_rate[active])
        self.joint_l1[active] = torch.nan_to_num(joint_l1[active])
        self.horizontal_speed[active] = torch.nan_to_num(horizontal_speed[active])
        self.joint_max[active] = torch.nan_to_num(joint_max[active])
        self.ankle_max[active] = torch.nan_to_num(ankle_max[active])
        self.foot_pitch_max[active] = torch.nan_to_num(foot_pitch_max[active])
        self.foot_roll_max[active] = torch.nan_to_num(foot_roll_max[active])
        self.tilt[active] = torch.acos(cos_tilt.clamp(-1.0, 1.0))[active]
        tracking = active & self.accepted & finite
        error = wrap_to_pi(self.target_yaw - current_yaw)
        self.heading_error[tracking] = error[tracking]
        self.max_heading_error[tracking] = torch.maximum(
            self.max_heading_error[tracking], error[tracking].abs()
        )
        safe = ~self.invalid & finite & (cos_tilt >= math.cos(math.radians(45)))
        if self.policy_version >= 7:
            repeated_ready = (safe & feet.all(-1) & ~body
                              & ((z - self.stand_z).abs() <= .01)
                              & (cos_tilt >= math.cos(math.radians(3.0)))
                              & (vz.abs() <= .05) & (horizontal_speed <= .03)
                              & (omega_xy.norm(dim=-1) <= .5)
                              & (yaw_rate.abs() <= .1) & (joint_l1 <= .08)
                              & (joint_max <= .10) & (ankle_max <= .06)
                              & (foot_pitch_max <= math.radians(3.0))
                              & (foot_roll_max <= math.radians(3.0)))
            # The strict gate is specifically the *repeat-jump* contract. V5/V6
            # source policies do not initially hold HOME within 0.08 rad, so
            # applying it before the first request would create a dead curriculum
            # with no jump observations or rewards.
            first_ready = (safe & feet.all(-1) & ~body
                           & ((z - self.stand_z).abs() <= .01)
                           & (cos_tilt >= math.cos(math.radians(15.0)))
                           & (vz.abs() <= .05) & (omega_xy.norm(dim=-1) <= .5))
            stable = torch.where(self.accepted, repeated_ready, first_ready)
        else:
            stable = (safe & feet.all(-1) & ((z - self.stand_z).abs() <= .01)
                      & (cos_tilt >= math.cos(math.radians(15))) & (vz.abs() <= .05)
                      & (omega_xy.norm(dim=-1) <= .5))
        self.ready_time[active] = torch.where(stable, self.ready_time + self.dt, 0)[active]
        self.request_time[active] += self.request[active] * self.dt
        # Both consecutive flight samples must themselves pass the validity gate.
        air = ~feet.any(-1) & self.busy & self.accepted & ~self.landed & safe
        self.air_steps[active] = torch.where(air, self.air_steps + 1, 0)[active]
        takeoff = active & (self.air_steps >= 2) & self.request & ~self.took_off
        self.took_off |= takeoff
        self.launch_signal[takeoff] = 1.
        drive = active & self.request & ~self.took_off & feet.all(-1) & safe
        frontier_v = torch.nan_to_num(vz / .8).clamp(0., 1.)
        self.launch_progress[drive] = .1 * (frontier_v - self.paid_launch).clamp_min(0)[drive]
        self.paid_launch[drive] = torch.maximum(self.paid_launch, frontier_v)[drive]
        expired = active & self.request & ~self.took_off & (self.request_time >= 1.5 - 1e-6)
        self.timeout_event[expired & ~self.request_timeout] = 1.
        self.timeout |= expired
        self.request_timeout |= expired
        self.request[active & (self.took_off | expired | self.invalid)] = False
        flight = active & self.took_off & air
        self.peak[flight] = torch.maximum(self.peak, z)[flight]
        frontier = ((self.peak - self.stand_z) / self.target_delta).clamp(0., 1.)
        self.height_progress[flight] = (frontier - self.paid_height).clamp_min(0)[flight]
        self.paid_height[flight] = torch.maximum(self.paid_height, frontier)[flight]
        touchdown = active & self.took_off & feet.any(-1) & ~self.impact_paid & finite
        impact = force * .01 + ((vz - self.prev_vz) / self.dt).abs() * .1
        self.impact[touchdown] = impact[touchdown]
        self.episode_impact[touchdown] += impact[touchdown]
        self.impact_paid |= touchdown
        self.landed |= active & self.took_off & feet.all(-1) & safe
        self.settle_hold |= active & self.took_off & feet.any(-1) & finite
        self.stable_time[active] = torch.where(self.landed & stable, self.stable_time + self.dt, 0)[active]
        if self.policy_version >= 8:
            recovering = active & self.landed & ~self.complete
            self.recovery_time[active] = torch.where(
                recovering, self.recovery_time + self.dt, self.recovery_time
            )[active]
            recovery_expired = (recovering & (self.recovery_time >= 8.0 - 1e-6)
                                & ~self.recovery_timeout)
            self.recovery_timeout_event[recovery_expired] = 1.0
            self.recovery_timeout |= recovery_expired
        goal = self.peak >= self.stand_z + self.acceptance_delta - 1e-6
        if self.policy_version >= 7:
            # There is no early V7 success payment. A request succeeds only
            # after the complete five-second recovery window.
            success = (active & goal
                       & (self.stable_time >= self.completion_duration - 1e-4)
                       & ~self.success & safe)
            complete = success & ~self.complete
        else:
            success = active & goal & (self.stable_time >= .25 - 1e-6) & ~self.success & safe
            complete = (active & goal
                        & (self.stable_time >= self.completion_duration - 1e-4)
                        & ~self.complete & safe)
            self.success_event[success] = 1.
        self.complete_event[complete] = 1.
        self.success |= success
        self.complete |= complete
        # No new request until the maneuver has completed, or a timed-out /
        # subtarget maneuver has returned through its versioned stability gate.
        release = ((self.landed & (self.stable_time >= self.completion_duration - 1e-4)) |
                   ((self.policy_version >= 8) & self.recovery_timeout) |
                   (self.request_timeout & (self.ready_time >= self.ready_duration - 1e-4)))
        self.busy[active & release] = False
        self.impact[active & self.fresh] = 0
        self.prev_vz[active] = vz[active]
        self.fresh[active] = False
        self.last_step[active] = step
