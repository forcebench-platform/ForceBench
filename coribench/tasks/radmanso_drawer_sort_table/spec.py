"""Simulator-free geometry contract for the RADMANSO drawer task."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random


DRAWER_JOINTS = {
    "upper": "PrismaticJoint_up",
    "middle": "PrismaticJoint_middle",
}
# The production task is intentionally fixed to the middle drawer.  The upper
# joint remains in DRAWER_JOINTS so the runtime can keep it locked.
SELECTABLE_DRAWERS = ("middle",)
BOTTOM_JOINT = "PrismaticJoint_down"

# Measured from the official ArtVIP USD at its closed pose.  The source drawer
# opens along local -Y and its authored joint coordinate ranges from -0.30 to 0.
HANDLE_LOCAL_M = {
    # All three handle Xforms are uniformly enlarged 2.5x in the task asset.
    # Scaling about each authored handle origin moves the collision/visual
    # center slightly farther out from the drawer front.
    "upper": (-0.0005263, -0.2530647, 0.6858543),
    "middle": (-0.0005263, -0.2530647, 0.4863018),
}
SOURCE_OPEN_LIMIT_M = 0.30


@dataclass(frozen=True)
class RadmansoDrawerSortTableSpec:
    asset_relative_path: str = (
        "external/artvip_drawers/radmanso_3_drawers/model_coribench.usda"
    )
    # At 45% scale the gold handle still protrudes 11.25 mm from the drawer
    # front, while the entire cabinet remains inside the canonical main view.
    cabinet_scale: float = 0.45
    # User-tuned placement keeps the cabinet and all three enlarged handles in
    # the main view while preserving clearance for approach and drawer travel.
    cabinet_position_m: tuple[float, float, float] = (0.620, -0.275, 0.0)
    cabinet_roll_rad: float = 0.0
    cabinet_pitch_rad: float = 0.0
    # Exactly 180 degrees: cabinet side edges are parallel/perpendicular to the
    # table edges, and local -Y opens along world +Y toward the robot/gripper.
    cabinet_yaw_rad: float = math.pi
    # Position-dependent resistance profile, expressed in world metres/newtons.
    # The drawer slides freely at first, enters a short rising-resistance band,
    # and irreversibly releases if the whole band is crossed.
    free_resistance_n: float = 0.35
    resistance_start_range_m: tuple[float, float] = (0.045, 0.090)
    resistance_band_length_range_m: tuple[float, float] = (0.012, 0.030)
    breakaway_resistance_range_n: tuple[float, float] = (5.5, 9.0)
    stop_resistance_n: float = 2.35
    post_breakaway_resistance_n: float = 0.0
    breakaway_hold_steps: int = 2
    success_tolerance_m: float = 0.0025
    success_stable_frames: int = 15
    # Independent safety stop. Even a failed expert that crosses the whole
    # resistance band cannot run the articulation to its 300 mm asset limit.
    max_safe_open_m: float = 0.135

    def handle_center_m(self, drawer: str) -> tuple[float, float, float]:
        if drawer not in HANDLE_LOCAL_M:
            raise ValueError(f"unsupported drawer: {drawer!r}")
        cx, cy, cz = self.cabinet_position_m
        lx, ly, lz = HANDLE_LOCAL_M[drawer]
        scale = float(self.cabinet_scale)
        cosine = math.cos(self.cabinet_yaw_rad)
        sine = math.sin(self.cabinet_yaw_rad)
        return (
            cx + scale * (cosine * lx - sine * ly),
            cy + scale * (sine * lx + cosine * ly),
            cz + scale * lz,
        )

    @property
    def top_handle_center_m(self) -> tuple[float, float, float]:
        return self.handle_center_m("upper")

    @property
    def middle_handle_center_m(self) -> tuple[float, float, float]:
        return self.handle_center_m("middle")

    @property
    def pull_direction_world(self) -> tuple[float, float, float]:
        return (
            math.sin(self.cabinet_yaw_rad),
            -math.cos(self.cabinet_yaw_rad),
            0.0,
        )

    @property
    def resistance_start_open_m(self) -> float:
        return 0.5 * sum(self.resistance_start_range_m)

    @property
    def resistance_band_length_m(self) -> float:
        return 0.5 * sum(self.resistance_band_length_range_m)

    @property
    def breakaway_open_m(self) -> float:
        return self.resistance_start_open_m + self.resistance_band_length_m

    @property
    def breakaway_resistance_n(self) -> float:
        return 0.5 * sum(self.breakaway_resistance_range_n)

    @property
    def success_open_m(self) -> float:
        return self.target_opening_for_force(
            resistance_start_open_m=self.resistance_start_open_m,
            breakaway_open_m=self.breakaway_open_m,
            breakaway_resistance_n=self.breakaway_resistance_n,
        )

    @property
    def success_joint_position_m(self) -> float:
        # Measured in simulation: uniform visual scaling does not scale the
        # prismatic displacement. One authored joint metre is one world metre.
        return -self.success_open_m

    def resistance_force_n(
        self,
        opening_world_m: float,
        *,
        resistance_start_open_m: float | None = None,
        breakaway_open_m: float | None = None,
        breakaway_resistance_n: float | None = None,
        broken: bool = False,
    ) -> float:
        """Designed pull resistance at one drawer opening.

        A cubic smoothstep avoids a solver impulse at the start of the high-load
        band while preserving the explicitly documented endpoints.
        """
        if broken:
            return float(self.post_breakaway_resistance_n)
        start = (
            self.resistance_start_open_m
            if resistance_start_open_m is None
            else float(resistance_start_open_m)
        )
        breakaway = (
            self.breakaway_open_m
            if breakaway_open_m is None
            else float(breakaway_open_m)
        )
        peak = (
            self.breakaway_resistance_n
            if breakaway_resistance_n is None
            else float(breakaway_resistance_n)
        )
        opening = max(0.0, float(opening_world_m))
        if opening <= start:
            return float(self.free_resistance_n)
        width = breakaway - start
        if width <= 0.0:
            raise ValueError("breakaway opening must exceed resistance start")
        u = min(1.0, (opening - start) / width)
        smooth = u * u * (3.0 - 2.0 * u)
        return float(
            self.free_resistance_n
            + smooth * (peak - self.free_resistance_n)
        )

    def target_opening_for_force(
        self,
        *,
        resistance_start_open_m: float,
        breakaway_open_m: float,
        breakaway_resistance_n: float,
    ) -> float:
        """Invert the monotonic resistance curve at the collection stop load."""
        start = float(resistance_start_open_m)
        breakaway = float(breakaway_open_m)
        peak = float(breakaway_resistance_n)
        if not self.free_resistance_n < self.stop_resistance_n < peak:
            raise ValueError("stop resistance must lie inside the rising band")
        low, high = start, breakaway
        for _ in range(48):
            middle = 0.5 * (low + high)
            force = self.resistance_force_n(
                middle,
                resistance_start_open_m=start,
                breakaway_open_m=breakaway,
                breakaway_resistance_n=peak,
            )
            if force < self.stop_resistance_n:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    def validate(self) -> None:
        if not 0.0 < self.cabinet_scale <= 1.0:
            raise ValueError("cabinet scale must be in (0, 1]")
        if abs(self.cabinet_roll_rad) > 1.0e-9 or abs(self.cabinet_pitch_rad) > 1.0e-9:
            raise ValueError("cabinet must stand upright on the table")
        if self.cabinet_position_m[2] < 0.0:
            raise ValueError("cabinet base would penetrate the tabletop")
        start_low, start_high = self.resistance_start_range_m
        length_low, length_high = self.resistance_band_length_range_m
        peak_low, peak_high = self.breakaway_resistance_range_n
        if not 0.02 <= start_low < start_high:
            raise ValueError("resistance start range must be positive and ordered")
        if not 0.0 < length_low < length_high:
            raise ValueError("resistance band length range must be positive and ordered")
        if start_high + length_high >= self.max_safe_open_m:
            raise ValueError("all resistance bands need clearance before the safety stop")
        if not self.max_safe_open_m < SOURCE_OPEN_LIMIT_M:
            raise ValueError("safety stop must stay inside the authored drawer limit")
        if not 0.0 <= self.free_resistance_n < peak_low < peak_high:
            raise ValueError("breakaway resistance must exceed free-travel resistance")
        if not self.free_resistance_n < self.stop_resistance_n < peak_low:
            raise ValueError("stop resistance must be reachable in every sampled band")
        if self.post_breakaway_resistance_n < 0.0:
            raise ValueError("post-breakaway resistance must be non-negative")
        if self.breakaway_hold_steps < 1 or self.success_stable_frames < 1:
            raise ValueError("breakaway and success hold counts must be positive")
        for start in (start_low, start_high):
            for length in (length_low, length_high):
                for peak in (peak_low, peak_high):
                    breakaway = start + length
                    target = self.target_opening_for_force(
                        resistance_start_open_m=start,
                        breakaway_open_m=breakaway,
                        breakaway_resistance_n=peak,
                    )
                    if not self.success_tolerance_m < min(
                        target - start, breakaway - target
                    ):
                        raise ValueError(
                            "success tolerance must stay inside every sampled band"
                        )


SPEC = RadmansoDrawerSortTableSpec()
SPEC.validate()


def sample_initial_state(seed: int) -> dict[str, object]:
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    selected_drawer = SELECTABLE_DRAWERS[seed % len(SELECTABLE_DRAWERS)]
    drawer_joint = DRAWER_JOINTS[selected_drawer]
    rng = random.Random(seed ^ 0x5A17D4A7)
    resistance_start = rng.uniform(*SPEC.resistance_start_range_m)
    resistance_band_length = rng.uniform(*SPEC.resistance_band_length_range_m)
    breakaway_open = resistance_start + resistance_band_length
    breakaway_resistance = rng.uniform(*SPEC.breakaway_resistance_range_n)
    success_open = SPEC.target_opening_for_force(
        resistance_start_open_m=resistance_start,
        breakaway_open_m=breakaway_open,
        breakaway_resistance_n=breakaway_resistance,
    )
    return {
        "seed": seed,
        "cabinet_position_m": list(SPEC.cabinet_position_m),
        "cabinet_roll_rad": SPEC.cabinet_roll_rad,
        "cabinet_pitch_rad": SPEC.cabinet_pitch_rad,
        "cabinet_yaw_rad": SPEC.cabinet_yaw_rad,
        "selected_drawer": selected_drawer,
        "handle_center_m": list(SPEC.handle_center_m(selected_drawer)),
        "top_handle_center_m": list(SPEC.top_handle_center_m),
        "middle_handle_center_m": list(SPEC.middle_handle_center_m),
        "pull_direction_world": list(SPEC.pull_direction_world),
        "drawer_joint": drawer_joint,
        "success_open_m": success_open,
        "success_joint_position_m": -success_open,
        "resistance_start_open_m": resistance_start,
        "resistance_band_length_m": resistance_band_length,
        "breakaway_open_m": breakaway_open,
        "free_resistance_n": SPEC.free_resistance_n,
        "breakaway_resistance_n": breakaway_resistance,
        "stop_resistance_n": SPEC.stop_resistance_n,
        "success_tolerance_m": SPEC.success_tolerance_m,
        "max_safe_open_m": SPEC.max_safe_open_m,
    }


def evaluate_final_state(
    drawer_position_m: float,
    selected_drawer: str = "middle",
    *,
    latch_broken: bool = False,
    stable_frames: int | None = None,
    peak_resistance_n: float | None = None,
    success_open_m: float | None = None,
    resistance_start_open_m: float | None = None,
    breakaway_open_m: float | None = None,
    breakaway_resistance_n: float | None = None,
) -> dict[str, object]:
    if selected_drawer not in DRAWER_JOINTS:
        raise ValueError(f"unsupported drawer: {selected_drawer!r}")
    position = float(drawer_position_m)
    opening_world_m = max(0.0, -position)
    target = SPEC.success_open_m if success_open_m is None else float(success_open_m)
    start = (
        SPEC.resistance_start_open_m
        if resistance_start_open_m is None
        else float(resistance_start_open_m)
    )
    breakaway = (
        SPEC.breakaway_open_m
        if breakaway_open_m is None
        else float(breakaway_open_m)
    )
    peak = (
        SPEC.breakaway_resistance_n
        if breakaway_resistance_n is None
        else float(breakaway_resistance_n)
    )
    position_error_m = abs(opening_world_m - target)
    within_target = position_error_m <= SPEC.success_tolerance_m
    stable = stable_frames is None or int(stable_frames) >= SPEC.success_stable_frames
    entered_resistance = (
        opening_world_m >= start
        if peak_resistance_n is None
        else float(peak_resistance_n) > SPEC.free_resistance_n + 0.25
    )
    success = bool(within_target and stable and entered_resistance and not latch_broken)
    return {
        "success": success,
        "drawer_open_enough": opening_world_m >= start,
        "within_target_window": within_target,
        "stable_at_target": stable,
        "entered_high_resistance": entered_resistance,
        "latch_broken": bool(latch_broken),
        "pulled_out": bool(latch_broken or opening_world_m >= breakaway),
        "selected_drawer": selected_drawer,
        "drawer_joint": DRAWER_JOINTS[selected_drawer],
        "drawer_position_m": position,
        "drawer_opening_world_m": opening_world_m,
        "position_error_m": position_error_m,
        "success_open_m": target,
        "resistance_start_open_m": start,
        "breakaway_open_m": breakaway,
        "breakaway_resistance_n": peak,
        "designed_resistance_n": SPEC.resistance_force_n(
            opening_world_m,
            resistance_start_open_m=start,
            breakaway_open_m=breakaway,
            breakaway_resistance_n=peak,
            broken=latch_broken,
        ),
        "peak_resistance_n": (
            None if peak_resistance_n is None else float(peak_resistance_n)
        ),
        "stable_frames": stable_frames,
        "spec": asdict(SPEC),
    }
