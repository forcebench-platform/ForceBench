"""Canonical arm-control contract and smooth joint trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


ARM_CONTROL_MODE = "smooth_position"


def shortest_angular_delta(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return the signed shortest angular displacement ``value - reference``.

    Revolute joint sensors may report the same physical angle on opposite sides
    of the ``-pi/+pi`` branch cut.  Computing the delta on the circle prevents a
    representation change from becoming a spurious approximately ``2*pi``
    motion.
    """

    value = torch.as_tensor(value)
    if not value.dtype.is_floating_point:
        value = value.to(dtype=torch.float32)
    reference = torch.as_tensor(reference, dtype=value.dtype, device=value.device)
    if value.shape != reference.shape:
        raise ValueError("angular value and reference must have the same shape")
    if not torch.isfinite(value).all() or not torch.isfinite(reference).all():
        raise ValueError("angular value and reference must be finite")
    delta = value - reference
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def unwrap_joint_positions(
    raw_position: torch.Tensor,
    previous_raw_position: torch.Tensor,
    previous_continuous_position: torch.Tensor,
) -> torch.Tensor:
    """Lift wrapped joint samples onto the continuous branch of the last sample."""

    raw_position = torch.as_tensor(raw_position)
    if not raw_position.dtype.is_floating_point:
        raw_position = raw_position.to(dtype=torch.float32)
    previous_raw_position = torch.as_tensor(
        previous_raw_position, dtype=raw_position.dtype, device=raw_position.device
    )
    previous_continuous_position = torch.as_tensor(
        previous_continuous_position, dtype=raw_position.dtype, device=raw_position.device
    )
    if not (
        raw_position.shape
        == previous_raw_position.shape
        == previous_continuous_position.shape
    ):
        raise ValueError("raw and continuous joint positions must have the same shape")
    return previous_continuous_position + shortest_angular_delta(
        raw_position, previous_raw_position
    )


@dataclass(frozen=True)
class JointServoLimits:
    """Limits used to time-scale a synchronized quintic joint trajectory."""

    max_velocity_rad_s: float = 1.5
    max_acceleration_rad_s2: float = 20.0
    max_jerk_rad_s3: float = 1500.0
    min_duration_s: float = 1.0 / 15.0
    constraint_samples: int = 65

    def validate(self) -> None:
        for name in (
            "max_velocity_rad_s",
            "max_acceleration_rad_s2",
            "max_jerk_rad_s3",
            "min_duration_s",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if int(self.constraint_samples) < 17:
            raise ValueError("constraint_samples must be at least 17")

    def to_dict(self) -> dict[str, float | int]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class SmoothPositionConfig:
    """Canonical legacy DLS IK and smooth joint-drive configuration."""

    actuator_stiffness: float = 80.0
    actuator_damping: float = 12.0
    dls_damping: float = 0.05
    max_position_error_m: float = 0.025
    max_rotation_error_rad: float = 0.12
    max_joint_step_rad: float = 0.08
    servo_limits: JointServoLimits = JointServoLimits()

    def validate(self) -> None:
        for name in (
            "actuator_stiffness",
            "actuator_damping",
            "dls_damping",
            "max_position_error_m",
            "max_rotation_error_rad",
            "max_joint_step_rad",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        self.servo_limits.validate()

    def profile(
        self, *, physics_rate_hz: int = 60, control_rate_hz: int = 15
    ) -> dict[str, object]:
        self.validate()
        physics_rate_hz = int(physics_rate_hz)
        control_rate_hz = int(control_rate_hz)
        if physics_rate_hz <= 0 or control_rate_hz <= 0 or physics_rate_hz % control_rate_hz:
            raise ValueError("physics rate must be a positive multiple of control rate")
        return {
            "mode": ARM_CONTROL_MODE,
            "implementation": "coribench.legacy_dls_smooth_position",
            "command_type": "pose",
            "use_relative_mode": False,
            "ik_method": "dls",
            "physics_rate_hz": physics_rate_hz,
            "control_rate_hz": control_rate_hz,
            "target_interpolation": "synchronized_quintic_joint_trajectory",
            "interpolation_steps": physics_rate_hz // control_rate_hz,
            "state_overwrite": False,
            "actuator_stiffness": self.actuator_stiffness,
            "actuator_damping": self.actuator_damping,
            "dls_damping": self.dls_damping,
            "max_position_error_m": self.max_position_error_m,
            "max_rotation_error_rad": self.max_rotation_error_rad,
            "max_joint_step_rad": self.max_joint_step_rad,
            "max_translation_command_m": self.max_position_error_m,
            "max_rotation_command_rad": self.max_rotation_error_rad,
            **self.servo_limits.to_dict(),
        }


class SynchronizedQuinticTrajectory:
    """Continuous joint target with finite velocity, acceleration and jerk."""

    _CONSTRAINT_MARGIN = 0.98
    _MAX_DURATION_ITERATIONS = 24

    def __init__(self, limits: JointServoLimits = JointServoLimits()):
        limits.validate()
        self.limits = limits
        self.position: torch.Tensor | None = None
        self.velocity: torch.Tensor | None = None
        self.acceleration: torch.Tensor | None = None
        self.jerk: torch.Tensor | None = None
        self.target: torch.Tensor | None = None
        self.duration_s = 0.0
        self.elapsed_s = 0.0
        self.active = False
        self._coefficients: torch.Tensor | None = None

    def reset(self, position: torch.Tensor) -> None:
        position = torch.as_tensor(position)
        if position.ndim not in (1, 2) or not torch.isfinite(position).all():
            raise ValueError("joint position must be a finite [D] or [B, D] tensor")
        if not position.dtype.is_floating_point:
            position = position.to(dtype=torch.float32)
        self.position = position.clone()
        self.velocity = torch.zeros_like(position)
        self.acceleration = torch.zeros_like(position)
        self.jerk = torch.zeros_like(position)
        self.target = position.clone()
        self.duration_s = 0.0
        self.elapsed_s = 0.0
        self.active = False
        self._coefficients = None

    def _require_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.position is None or self.velocity is None or self.acceleration is None:
            raise RuntimeError("trajectory must be reset before use")
        return self.position, self.velocity, self.acceleration

    def _initial_duration(self, target: torch.Tensor, minimum: float) -> float:
        position, _, _ = self._require_state()
        distance = torch.max(torch.abs(target - position)).item()
        velocity_time = 1.875 * distance / self.limits.max_velocity_rad_s
        acceleration_time = (5.78 * distance / self.limits.max_acceleration_rad_s2) ** 0.5
        jerk_time = (60.0 * distance / self.limits.max_jerk_rad_s3) ** (1.0 / 3.0)
        return max(float(minimum), velocity_time, acceleration_time, jerk_time, 1.0e-4)

    def _make_coefficients(self, target: torch.Tensor, duration: float) -> torch.Tensor:
        position, velocity, acceleration = self._require_state()
        displacement = target - position
        c0 = position
        c1 = velocity
        c2 = 0.5 * acceleration
        c3 = (
            20.0 * displacement - 12.0 * velocity * duration - 3.0 * acceleration * duration**2
        ) / (2.0 * duration**3)
        c4 = (
            -30.0 * displacement + 16.0 * velocity * duration + 3.0 * acceleration * duration**2
        ) / (2.0 * duration**4)
        c5 = (
            12.0 * displacement - 6.0 * velocity * duration - acceleration * duration**2
        ) / (2.0 * duration**5)
        return torch.stack((c0, c1, c2, c3, c4, c5), dim=0)

    @staticmethod
    def _evaluate(coefficients: torch.Tensor, time_s: torch.Tensor | float):
        time = torch.as_tensor(time_s, dtype=coefficients.dtype, device=coefficients.device)
        if time.ndim:
            for _ in range(coefficients.ndim - 1):
                time = time.unsqueeze(-1)
        c0, c1, c2, c3, c4, c5 = coefficients
        position = c0 + time * (c1 + time * (c2 + time * (c3 + time * (c4 + time * c5))))
        velocity = c1 + time * (2.0 * c2 + time * (3.0 * c3 + time * (4.0 * c4 + time * 5.0 * c5)))
        acceleration = 2.0 * c2 + time * (6.0 * c3 + time * (12.0 * c4 + time * 20.0 * c5))
        jerk = 6.0 * c3 + time * (24.0 * c4 + time * 60.0 * c5)
        return position, velocity, acceleration, jerk

    def _constraint_scale(self, coefficients: torch.Tensor, duration: float) -> float:
        samples = torch.linspace(
            0.0,
            float(duration),
            int(self.limits.constraint_samples),
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        _, velocity, acceleration, jerk = self._evaluate(coefficients, samples)
        velocity_ratio = torch.max(torch.abs(velocity)).item() / self.limits.max_velocity_rad_s
        acceleration_ratio = (
            torch.max(torch.abs(acceleration)).item() / self.limits.max_acceleration_rad_s2
        ) ** 0.5
        jerk_ratio = (
            torch.max(torch.abs(jerk)).item() / self.limits.max_jerk_rad_s3
        ) ** (1.0 / 3.0)
        return max(velocity_ratio, acceleration_ratio, jerk_ratio)

    def plan(self, target: torch.Tensor, *, min_duration_s: float | None = None) -> bool:
        position, velocity, acceleration = self._require_state()
        target = torch.as_tensor(target, dtype=position.dtype, device=position.device)
        if target.shape != position.shape or not torch.isfinite(target).all():
            raise ValueError(f"joint target must be finite with shape {tuple(position.shape)}")
        if self.target is not None and torch.allclose(target, self.target, rtol=0.0, atol=1.0e-7):
            return False
        self.target = target.clone()
        if (
            torch.allclose(target, position, rtol=0.0, atol=1.0e-7)
            and torch.max(torch.abs(velocity)).item() < 1.0e-7
            and torch.max(torch.abs(acceleration)).item() < 1.0e-6
        ):
            self.reset(target)
            return True
        minimum = self.limits.min_duration_s if min_duration_s is None else float(min_duration_s)
        if minimum <= 0.0:
            raise ValueError("min_duration_s must be positive")
        duration = self._initial_duration(target, minimum)
        coefficients = None
        for _ in range(self._MAX_DURATION_ITERATIONS):
            coefficients = self._make_coefficients(target, duration)
            scale = self._constraint_scale(coefficients, duration)
            if scale <= self._CONSTRAINT_MARGIN:
                break
            duration *= max(1.05, 1.02 * scale / self._CONSTRAINT_MARGIN)
        else:
            raise RuntimeError("could not construct a bounded quintic joint trajectory")
        self._coefficients = coefficients
        self.duration_s = float(duration)
        self.elapsed_s = 0.0
        self.active = True
        return True

    def step(self, dt: float) -> torch.Tensor:
        position, _, _ = self._require_state()
        if float(dt) <= 0.0:
            raise ValueError("trajectory step dt must be positive")
        if not self.active:
            return position
        if self._coefficients is None or self.target is None:
            raise RuntimeError("active trajectory has no coefficients or target")
        self.elapsed_s = min(self.elapsed_s + float(dt), self.duration_s)
        position, velocity, acceleration, jerk = self._evaluate(
            self._coefficients, self.elapsed_s
        )
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration
        self.jerk = jerk
        if self.elapsed_s >= self.duration_s:
            self.position = self.target.clone()
            self.velocity = torch.zeros_like(self.position)
            self.acceleration = torch.zeros_like(self.position)
            self.jerk = torch.zeros_like(self.position)
            self.active = False
        return self.position


DEFAULT_SMOOTH_POSITION_CONFIG = SmoothPositionConfig()
DEFAULT_SMOOTH_POSITION_CONFIG.validate()
DEFAULT_SERVO_LIMITS = DEFAULT_SMOOTH_POSITION_CONFIG.servo_limits
DEFAULT_ARM_CONTROLLER_PROFILE = DEFAULT_SMOOTH_POSITION_CONFIG.profile()


__all__ = [
    "ARM_CONTROL_MODE",
    "DEFAULT_ARM_CONTROLLER_PROFILE",
    "DEFAULT_SMOOTH_POSITION_CONFIG",
    "DEFAULT_SERVO_LIMITS",
    "SmoothPositionConfig",
    "JointServoLimits",
    "SynchronizedQuinticTrajectory",
    "shortest_angular_delta",
    "unwrap_joint_positions",
]
