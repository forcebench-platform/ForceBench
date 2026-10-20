"""Per-clone scheduling of the EXISTING stove expert's stages and gates.

No new thresholds, reverse correction, or physics/controller changes.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class CompletionSnapshot:
    """Post-action state for the final retained transition of one clone.

    Peer clones may continue simulating after this point. Collection quality
    gates must use this snapshot, rather than their later shared batch state.
    """

    report: dict
    control_frames: int
    press_m: float
    turn_angle_rad: float
    fingertip_xyz_m: tuple[float, float, float]
    history_metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def capture(cls, *, report, control_frames, press_m, turn_angle_rad, fingertip_xyz_m,
                history_metrics=None):
        point = tuple(float(value) for value in fingertip_xyz_m)
        values = (*point, float(press_m), float(turn_angle_rad))
        if len(point) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError("completion state must have finite mechanism and fingertip values")
        if int(control_frames) < 1:
            raise ValueError("completion state must retain at least one transition")
        return cls(
            deepcopy(report), int(control_frames), float(press_m), float(turn_angle_rad), point,
            {key: float(value) for key, value in (history_metrics or {}).items()},
        )


@dataclass
class ReleaseState:
    target_deg: float
    easy_sign: int
    tolerance_deg: float
    profile: dict
    stage: str = "turn"
    frames: int = 0
    stable_count: int = 0
    settle_complete: bool = False
    release_passed: bool = False
    release_frames: int = 0
    release_range_deg: float = float("inf")
    failure_reason: str = ""
    window: deque = field(default_factory=deque)

    def __post_init__(self):
        self.window = deque(maxlen=self.profile["release_stability_window_frames"])

    def transition(self, stage):
        self.stage, self.frames = stage, 0

    def prepare(self, angle_deg):
        """Existing pre-step settle check and forward-only 0.08-deg adjustment."""
        if self.stage != "settle":
            return 0.0
        if self.frames >= self.profile["maximum_settle_frames"]:
            self.transition("release_check")
            return 0.0
        error = self.target_deg - angle_deg
        self.stable_count = self.stable_count + 1 if abs(error) <= self.profile["settle_tolerance_deg"] else 0
        self.settle_complete |= self.stable_count >= self.profile["settle_stable_frames"]
        if self.settle_complete:
            self.transition("release_check")
            return 0.0
        directed_error = self.easy_sign * error
        return self.easy_sign * min(max(directed_error, 0.0), self.profile["settle_command_step_deg"])

    def observe(self, angle_deg, *, turn_complete=False):
        """Advance THIS clone after one physical control step; never wait for peers."""
        self.frames += 1
        if self.stage == "turn":
            if turn_complete or self.frames >= self.profile["maximum_turn_frames"]:
                self.transition("settle")
        elif self.stage == "release_check":
            self.window.append(float(angle_deg))
            if len(self.window) == self.window.maxlen:
                values = list(self.window)
                span = max(values) - min(values)
                mean_step = sum(abs(b-a) for a,b in zip(values,values[1:])) / (len(values)-1)
                target_error = abs(self.target_deg-angle_deg)
                if (span <= self.profile["release_stability_range_deg"]
                        and mean_step <= self.profile["release_stability_mean_step_deg"]
                        and target_error <= self.tolerance_deg):
                    self.release_passed = True
                    self.release_frames, self.release_range_deg = self.frames, span
                    self.transition("open_ramp")
                    return
            if self.frames >= self.profile["release_stability_max_frames"]:
                self.release_frames = self.frames
                self.failure_reason = "target_out_of_range" if abs(self.target_deg-angle_deg) > self.tolerance_deg else "motion_not_stable"
                self.transition("failed")
        else:
            sequence = {
                "open_ramp": ("open_ramp_frames", "open_hold"),
                "open_hold": ("open_hold_frames", "release_press"),
                "release_press": ("release_press_frames", "lift"),
                "lift": ("lift_frames", "retreat"),
                "retreat": ("retreat_frames", "done"),
            }
            if self.stage in sequence:
                key, next_stage = sequence[self.stage]
                if self.frames >= self.profile[key]:
                    self.transition(next_stage)

    @property
    def terminal(self):
        return self.stage in {"done", "failed"}


def trim_episode_buffer(buffer, count):
    """Trim only post-completion padding; preserve control/physics alignment."""
    from dataclasses import fields
    if not 0 < count <= len(buffer):
        raise ValueError(f"invalid episode length: {count}/{len(buffer)}")
    for field in fields(buffer):
        value = getattr(buffer, field.name)
        if isinstance(value, list):
            del value[count:]
        elif field.name == "privileged_fields":
            for series in value.values():
                del series[count:]
