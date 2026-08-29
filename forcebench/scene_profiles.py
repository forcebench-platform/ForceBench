"""Benchmark-level visual scene profiles.

The task plugins own manipulands and success criteria.  Scene profiles only
own the surrounding room, support-table appearance and lighting, so a new
task can reuse the same visual identity without inheriting peg-insertion
assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


DEFAULT_SCENE_PROFILE = "contact_lab"


@dataclass(frozen=True)
class VisualBox:
    """A collision-free architectural or furniture accent."""

    name: str
    size_m: tuple[float, float, float]
    translation_m: tuple[float, float, float]
    color: tuple[float, float, float]
    roughness: float = 0.65
    metallic: float = 0.0
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"invalid visual box name: {self.name!r}")
        if any(value <= 0.0 for value in self.size_m):
            raise ValueError(f"visual box {self.name!r} must have positive dimensions")
        if any(value < 0.0 or value > 1.0 for value in self.color):
            raise ValueError(f"visual box {self.name!r} color must be in [0, 1]")
        if self.roughness < 0.0 or self.roughness > 1.0:
            raise ValueError(f"visual box {self.name!r} roughness must be in [0, 1]")
        if self.metallic < 0.0 or self.metallic > 1.0:
            raise ValueError(f"visual box {self.name!r} metallic must be in [0, 1]")
        if any(value < 0.0 for value in self.emissive_color):
            raise ValueError(f"visual box {self.name!r} emissive color must be non-negative")


@dataclass(frozen=True)
class SceneProfile:
    """Visual shell around the invariant ForceBench physics workspace."""

    name: str
    description: str
    room_usd_path: str | None
    room_translation_m: tuple[float, float, float]
    room_orientation_xyzw: tuple[float, float, float, float]
    hidden_room_prims: tuple[str, ...]
    visual_boxes: tuple[VisualBox, ...]
    table_visual_usd_path: str | None
    table_visual_translation_m: tuple[float, float, float]
    table_visual_orientation_xyzw: tuple[float, float, float, float]
    table_visual_scale: tuple[float, float, float]
    table_visual_collision_prim: str | None
    table_size_m: tuple[float, float, float]
    table_translation_m: tuple[float, float, float]
    ground_center_z_m: float
    ground_visible: bool
    ground_color: tuple[float, float, float]
    table_visible: bool
    table_color: tuple[float, float, float]
    table_roughness: float
    table_metallic: float
    dome_light_intensity: float
    dome_light_color: tuple[float, float, float]
    room_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    evaluation_num_envs: int = 100

    def validate(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"invalid scene profile name: {self.name!r}")
        if self.table_visual_usd_path is None and not self.table_visible:
            raise ValueError("a scene profile must expose either the native or USD table visual")
        if any(value <= 0.0 for value in self.table_visual_scale):
            raise ValueError("table visual scale values must be positive")
        if sum(value * value for value in self.table_visual_orientation_xyzw) <= 0.0:
            raise ValueError("table visual orientation must be a non-zero quaternion")
        if self.table_visual_collision_prim is not None and self.table_visual_usd_path is None:
            raise ValueError("a table collision branch requires a table visual USD")
        if any(value <= 0.0 for value in self.table_size_m):
            raise ValueError("table size values must be positive")
        if self.table_roughness < 0.0 or self.table_roughness > 1.0:
            raise ValueError("table roughness must be in [0, 1]")
        if self.table_metallic < 0.0 or self.table_metallic > 1.0:
            raise ValueError("table metallic must be in [0, 1]")
        if self.dome_light_intensity < 0.0:
            raise ValueError("dome light intensity must be non-negative")
        if any(value <= 0.0 for value in self.room_scale):
            raise ValueError("room scale values must be positive")
        if not 1 <= self.evaluation_num_envs <= 100:
            raise ValueError("evaluation_num_envs must be in [1, 100]")
        for box in self.visual_boxes:
            box.validate()


_CONTACT_LAB = SceneProfile(
    name="contact_lab",
    description="Warm ForceBench study workshop with an exact-height light-wood worktable.",
    room_usd_path=None,
    room_translation_m=(0.0, 0.0, 0.0),
    room_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    hidden_room_prims=(),
    # LIBERO-inspired visual language without importing LIBERO assets or
    # runtime code: light wood, warm plaster, simple study furniture, and an
    # uncluttered task surface.  Every item remains appearance-only.  The floor
    # meets the worktable legs at z=-0.7947 m.
    visual_boxes=(
        VisualBox("floor", (5.0, 5.0, 0.04), (0.55, 0.0, -0.8147), (0.19, 0.175, 0.15), roughness=0.82),
        VisualBox("rear_wall", (0.08, 3.20, 2.40), (-1.58, 0.0, 0.43), (0.19, 0.18, 0.165), roughness=0.90),
        VisualBox("side_wall_left", (4.20, 0.08, 2.40), (0.55, 1.48, 0.43), (0.16, 0.165, 0.16), roughness=0.90),
        VisualBox("side_wall_right", (4.20, 0.08, 2.40), (0.55, -1.48, 0.43), (0.16, 0.165, 0.16), roughness=0.90),
        VisualBox("rear_wainscot", (0.035, 3.20, 0.44), (-1.535, 0.0, -0.50), (0.105, 0.095, 0.080), roughness=0.68),
        # A quiet study-like backdrop gives the benchmark a recognizable room
        # identity while leaving the manipulation surface visually sparse.
        VisualBox("wall_art", (0.030, 0.70, 0.46), (-1.525, 0.0, 0.62), (0.095, 0.155, 0.145), roughness=0.62),
        VisualBox("cabinet_left", (0.20, 0.50, 0.86), (-1.43, 0.98, -0.25), (0.070, 0.060, 0.050), roughness=0.52),
        VisualBox("cabinet_right", (0.20, 0.50, 0.86), (-1.43, -0.98, -0.25), (0.070, 0.060, 0.050), roughness=0.52),
        VisualBox("cabinet_top_left", (0.24, 0.54, 0.045), (-1.40, 0.98, 0.2025), (0.36, 0.245, 0.145), roughness=0.70),
        VisualBox("cabinet_top_right", (0.24, 0.54, 0.045), (-1.40, -0.98, 0.2025), (0.36, 0.245, 0.145), roughness=0.70),
        VisualBox("cabinet_handle_left", (0.018, 0.16, 0.018), (-1.322, 0.98, -0.10), (0.29, 0.27, 0.23), roughness=0.24, metallic=0.62),
        VisualBox("cabinet_handle_right", (0.018, 0.16, 0.018), (-1.322, -0.98, -0.10), (0.29, 0.27, 0.23), roughness=0.24, metallic=0.62),
        # Dark wood frame and legs stop at z=-0.025 m, directly beneath
        # the visible/colliding tabletop.  They are appearance-only, so task
        # dynamics remain defined by the single exact-height tabletop cube.
        VisualBox("bench_front_apron", (1.39, 0.045, 0.080), (0.475, -0.4225, -0.065), (0.075, 0.055, 0.040), roughness=0.52),
        VisualBox("bench_back_apron", (1.39, 0.045, 0.080), (0.475, 0.4225, -0.065), (0.075, 0.055, 0.040), roughness=0.52),
        VisualBox("bench_left_apron", (0.045, 0.80, 0.080), (-0.2225, 0.0, -0.065), (0.075, 0.055, 0.040), roughness=0.52),
        VisualBox("bench_right_apron", (0.045, 0.80, 0.080), (1.1725, 0.0, -0.065), (0.075, 0.055, 0.040), roughness=0.52),
        VisualBox("bench_leg_front_left", (0.050, 0.050, 0.730), (-0.165, -0.365, -0.415), (0.085, 0.060, 0.042), roughness=0.48),
        VisualBox("bench_leg_front_right", (0.050, 0.050, 0.730), (1.115, -0.365, -0.415), (0.085, 0.060, 0.042), roughness=0.48),
        VisualBox("bench_leg_back_left", (0.050, 0.050, 0.730), (-0.165, 0.365, -0.415), (0.085, 0.060, 0.042), roughness=0.48),
        VisualBox("bench_leg_back_right", (0.050, 0.050, 0.730), (1.115, 0.365, -0.415), (0.085, 0.060, 0.042), roughness=0.48),
        VisualBox("bench_lower_front", (1.33, 0.035, 0.045), (0.475, -0.365, -0.690), (0.10, 0.072, 0.050), roughness=0.50),
        VisualBox("bench_lower_back", (1.33, 0.035, 0.045), (0.475, 0.365, -0.690), (0.10, 0.072, 0.050), roughness=0.50),
    ),
    table_visual_usd_path=None,
    table_visual_translation_m=(0.0, 0.0, 0.0),
    table_visual_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    table_visual_scale=(1.0, 1.0, 1.0),
    table_visual_collision_prim=None,
    # Extend the top to x=-0.20 m so the Franka link0 visual bound
    # (x_min=-0.1541 m) is fully supported instead of overhanging the bench.
    table_size_m=(1.35, 0.80, 0.05),
    table_translation_m=(0.475, 0.0, -0.025),
    ground_center_z_m=-0.8447,
    ground_visible=False,
    ground_color=(0.12, 0.14, 0.17),
    table_visible=True,
    # Warm, matte light wood follows LIBERO's readable tabletop style while
    # retaining our exact-height native collider and saturated task fixtures.
    table_color=(0.42, 0.30, 0.19),
    table_roughness=0.64,
    table_metallic=0.0,
    dome_light_intensity=1450.0,
    dome_light_color=(0.96, 0.91, 0.84),
)

_WEIGHT_SORT_RESIDENTIAL = SceneProfile(
    name="weight_sort_residential",
    description=(
        "Weight-sort-only residential scene with the EastRural dining table "
        "oriented with its long edge across the main camera view."
    ),
    room_usd_path=None,
    room_translation_m=(0.0, 0.0, 0.0),
    room_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    hidden_room_prims=(),
    # Reuse the contact-lab room shell, but remove its synthetic bench frame
    # and legs because the EastRural asset supplies the complete table.
    visual_boxes=tuple(
        box for box in _CONTACT_LAB.visual_boxes if not box.name.startswith("bench_")
    ),
    table_visual_usd_path=(
        "forcebench://assets/residential/east_rural/EastRural_Table_weight_sort.usda"
    ),
    # EastRural is natively 0.9144 x 1.5273 x 0.7620 m.  Its long axis is
    # already world Y, which is horizontal in the unchanged main camera.
    table_visual_translation_m=(0.40, 0.0, -0.72),
    table_visual_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    # The source asset is authored in centimeters (metersPerUnit=0.01), while
    # ForceBench's live stage uses meters.  Include that unit conversion here.
    table_visual_scale=(0.01202975, 0.01014864, 0.00944882),
    table_visual_collision_prim="collisions",
    table_size_m=(1.10, 1.55, 0.05),
    table_translation_m=(0.40, 0.0, -0.025),
    ground_center_z_m=-0.77,
    ground_visible=False,
    ground_color=(0.19, 0.175, 0.15),
    # Keep one exact, invisible ForceBench collider at z=0.  The imported
    # residential asset is appearance-only.
    table_visible=False,
    table_color=(0.42, 0.30, 0.19),
    table_roughness=0.64,
    table_metallic=0.0,
    dome_light_intensity=1450.0,
    dome_light_color=(0.96, 0.91, 0.84),
)

_PRECISION_ASH_VENEER = replace(
    _WEIGHT_SORT_RESIDENTIAL,
    name="precision_ash_veneer",
    description=(
        "Poly Haven ash veneer PBR tabletop for precision insertion "
        "tasks, retaining the EastRural geometry and native task collider."
    ),
    table_visual_usd_path=(
        "forcebench://assets/residential/east_rural/"
        "EastRural_Table_ash_veneer_precision.usda"
    ),
)

_PARCEL_SORT_STATION = SceneProfile(
    name="parcel_sort_station",
    description=(
        "Parcel-sorting station built around the official NVIDIA small-warehouse "
        "rack while preserving the ForceBench robot base and camera geometry."
    ),
    room_usd_path=(
        "forcebench://assets/simready/small_warehouse_digital_twin/rack.usd"
    ),
    # The 65%-scale official rack now sits behind the robot as warehouse
    # background instead of wrapping around the worktable.
    room_translation_m=(0.10, 1.65, -0.7947),
    room_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    room_scale=(0.65, 0.65, 0.65),
    hidden_room_prims=(),
    visual_boxes=(
        VisualBox("parcel_floor", (5.0, 5.0, 0.04), (0.48, 0.0, -0.8147), (0.24, 0.25, 0.25), roughness=0.88),
        VisualBox("parcel_rear_wall", (0.08, 3.20, 2.40), (-1.58, 0.0, 0.43), (0.45, 0.47, 0.48), roughness=0.84),
        # Positive world Y is the conveyor-outfeed side; leave it open like a
        # real sorting cell instead of sending the belt through a wall.
        VisualBox("parcel_right_wall", (4.20, 0.08, 2.40), (0.48, -1.48, 0.43), (0.37, 0.39, 0.40), roughness=0.86),
        VisualBox("safety_band", (0.035, 2.40, 0.16), (-1.535, 0.0, -0.46), (0.92, 0.56, 0.05), roughness=0.58),
        VisualBox("station_sign", (0.025, 0.90, 0.34), (-1.525, 0.0, 0.67), (0.08, 0.22, 0.32), roughness=0.44),
        VisualBox("station_sign_bar", (0.020, 0.62, 0.055), (-1.505, 0.0, 0.67), (0.94, 0.62, 0.08), roughness=0.50),
        VisualBox("bollard_left", (0.10, 0.10, 0.62), (-0.52, -1.05, -0.49), (0.95, 0.62, 0.04), roughness=0.48, metallic=0.34),
        VisualBox("bollard_right", (0.10, 0.10, 0.62), (-0.52, 1.05, -0.49), (0.95, 0.62, 0.04), roughness=0.48, metallic=0.34),
        VisualBox("parcel_bench_front", (1.39, 0.045, 0.080), (0.475, -0.4225, -0.065), (0.10, 0.12, 0.13), roughness=0.40, metallic=0.62),
        VisualBox("parcel_bench_back", (1.39, 0.045, 0.080), (0.475, 0.4225, -0.065), (0.10, 0.12, 0.13), roughness=0.40, metallic=0.62),
        VisualBox("parcel_bench_leg_fl", (0.060, 0.060, 0.730), (-0.165, -0.365, -0.415), (0.10, 0.12, 0.13), roughness=0.40, metallic=0.62),
        VisualBox("parcel_bench_leg_fr", (0.060, 0.060, 0.730), (1.115, -0.365, -0.415), (0.10, 0.12, 0.13), roughness=0.40, metallic=0.62),
        VisualBox("parcel_bench_leg_bl", (0.060, 0.060, 0.730), (-0.165, 0.365, -0.415), (0.10, 0.12, 0.13), roughness=0.40, metallic=0.62),
        VisualBox("parcel_bench_leg_br", (0.060, 0.060, 0.730), (1.115, 0.365, -0.415), (0.10, 0.12, 0.13), roughness=0.40, metallic=0.62),
        VisualBox("ceiling_light_left", (1.20, 0.12, 0.035), (-0.30, -0.72, 1.42), (0.72, 0.75, 0.76), roughness=0.28, emissive_color=(4.0, 4.2, 4.3)),
        VisualBox("ceiling_light_right", (1.20, 0.12, 0.035), (-0.30, 0.72, 1.42), (0.72, 0.75, 0.76), roughness=0.28, emissive_color=(4.0, 4.2, 4.3)),
    ),
    table_visual_usd_path=None,
    table_visual_translation_m=(0.0, 0.0, 0.0),
    table_visual_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    table_visual_scale=(1.0, 1.0, 1.0),
    table_visual_collision_prim=None,
    table_size_m=(1.35, 0.80, 0.05),
    table_translation_m=(0.475, 0.0, -0.025),
    ground_center_z_m=-0.8447,
    ground_visible=False,
    ground_color=(0.24, 0.25, 0.25),
    table_visible=True,
    table_color=(0.43, 0.46, 0.47),
    table_roughness=0.46,
    table_metallic=0.12,
    dome_light_intensity=1750.0,
    dome_light_color=(0.92, 0.96, 1.0),
)

# Exact visual shell used by the 2026-08-23 Rattle expert dataset.  The task
# geometry, table, lighting, cameras, and synthetic station accents are
# unchanged; only the later-added warehouse rack background is absent.
_PARCEL_SORT_STATION_COLLECTION_20260823 = replace(
    _PARCEL_SORT_STATION,
    name="parcel_sort_station_collection_20260823",
    description=(
        "Rattle expert-data scene from 2026-08-23, before the warehouse rack "
        "was added as an appearance-only background."
    ),
    room_usd_path=None,
)

_GEAR_ASSEMBLY_CELL = SceneProfile(
    name="gear_assembly_cell",
    description=(
        "Compact industrial gear-assembly cell using the NVIDIA small-warehouse "
        "rack as collision-free background around a clean metal workbench."
    ),
    room_usd_path=(
        "forcebench://assets/simready/small_warehouse_digital_twin/rack.usd"
    ),
    # Keep the rack behind and slightly to the side of the robot so it reads as
    # factory context without occluding the tabletop, gripper, or gears.
    room_translation_m=(-0.10, 1.62, -0.7947),
    room_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    room_scale=(0.58, 0.58, 0.58),
    hidden_room_prims=(),
    visual_boxes=(
        VisualBox("gear_floor", (5.0, 5.0, 0.04), (0.48, 0.0, -0.8147), (0.18, 0.20, 0.21), roughness=0.88),
        VisualBox("gear_rear_wall", (0.08, 3.20, 2.40), (-1.58, 0.0, 0.43), (0.31, 0.34, 0.35), roughness=0.82),
        VisualBox("gear_left_wall", (4.20, 0.08, 2.40), (0.48, 1.48, 0.43), (0.25, 0.28, 0.29), roughness=0.84),
        VisualBox("gear_right_wall", (4.20, 0.08, 2.40), (0.48, -1.48, 0.43), (0.25, 0.28, 0.29), roughness=0.84),
        # Structural steel and cool overhead strip lights create the local
        # factory-cell silhouette visible from the fixed main camera.
        VisualBox("gear_beam_rear", (0.16, 3.10, 0.16), (-1.46, 0.0, 1.43), (0.075, 0.09, 0.10), roughness=0.34, metallic=0.70),
        VisualBox("gear_beam_left", (2.95, 0.14, 0.14), (-0.05, 1.34, 1.43), (0.075, 0.09, 0.10), roughness=0.34, metallic=0.70),
        VisualBox("gear_beam_right", (2.95, 0.14, 0.14), (-0.05, -1.34, 1.43), (0.075, 0.09, 0.10), roughness=0.34, metallic=0.70),
        VisualBox("gear_light_left", (1.25, 0.13, 0.035), (-0.25, -0.68, 1.37), (0.78, 0.82, 0.84), roughness=0.24, emissive_color=(5.0, 5.2, 5.4)),
        VisualBox("gear_light_right", (1.25, 0.13, 0.035), (-0.25, 0.68, 1.37), (0.78, 0.82, 0.84), roughness=0.24, emissive_color=(5.0, 5.2, 5.4)),
        # Two low tool cabinets frame the station while remaining well outside
        # the robot and task-object workspace.
        VisualBox("gear_tool_cabinet_left", (0.30, 0.54, 0.82), (-1.36, -0.92, -0.38), (0.055, 0.12, 0.16), roughness=0.42, metallic=0.48),
        VisualBox("gear_tool_cabinet_right", (0.30, 0.54, 0.82), (-1.36, 0.92, -0.38), (0.055, 0.12, 0.16), roughness=0.42, metallic=0.48),
        VisualBox("gear_cabinet_top_left", (0.34, 0.58, 0.045), (-1.34, -0.92, 0.052), (0.24, 0.27, 0.28), roughness=0.30, metallic=0.68),
        VisualBox("gear_cabinet_top_right", (0.34, 0.58, 0.045), (-1.34, 0.92, 0.052), (0.24, 0.27, 0.28), roughness=0.30, metallic=0.68),
        VisualBox("gear_handle_left", (0.020, 0.20, 0.020), (-1.198, -0.92, -0.30), (0.62, 0.66, 0.67), roughness=0.20, metallic=0.82),
        VisualBox("gear_handle_right", (0.020, 0.20, 0.020), (-1.198, 0.92, -0.30), (0.62, 0.66, 0.67), roughness=0.20, metallic=0.82),
        VisualBox("gear_tool_board", (0.035, 1.15, 0.48), (-1.525, 0.0, 0.62), (0.07, 0.16, 0.20), roughness=0.48, metallic=0.32),
        VisualBox("gear_tool_board_band", (0.020, 0.82, 0.055), (-1.503, 0.0, 0.62), (0.93, 0.58, 0.04), roughness=0.48),
        # Appearance-only steel frame under the unchanged exact-height native
        # tabletop collider.
        VisualBox("gear_bench_front", (1.39, 0.055, 0.10), (0.475, -0.4225, -0.075), (0.075, 0.09, 0.10), roughness=0.32, metallic=0.72),
        VisualBox("gear_bench_back", (1.39, 0.055, 0.10), (0.475, 0.4225, -0.075), (0.075, 0.09, 0.10), roughness=0.32, metallic=0.72),
        VisualBox("gear_bench_leg_fl", (0.065, 0.065, 0.73), (-0.165, -0.365, -0.415), (0.075, 0.09, 0.10), roughness=0.32, metallic=0.72),
        VisualBox("gear_bench_leg_fr", (0.065, 0.065, 0.73), (1.115, -0.365, -0.415), (0.075, 0.09, 0.10), roughness=0.32, metallic=0.72),
        VisualBox("gear_bench_leg_bl", (0.065, 0.065, 0.73), (-0.165, 0.365, -0.415), (0.075, 0.09, 0.10), roughness=0.32, metallic=0.72),
        VisualBox("gear_bench_leg_br", (0.065, 0.065, 0.73), (1.115, 0.365, -0.415), (0.075, 0.09, 0.10), roughness=0.32, metallic=0.72),
        VisualBox("gear_bench_lower_shelf", (1.20, 0.62, 0.045), (0.475, 0.0, -0.65), (0.12, 0.14, 0.15), roughness=0.40, metallic=0.62),
    ),
    table_visual_usd_path=None,
    table_visual_translation_m=(0.0, 0.0, 0.0),
    table_visual_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    table_visual_scale=(1.0, 1.0, 1.0),
    table_visual_collision_prim=None,
    table_size_m=(1.35, 0.80, 0.05),
    table_translation_m=(0.475, 0.0, -0.025),
    ground_center_z_m=-0.8447,
    ground_visible=False,
    ground_color=(0.18, 0.20, 0.21),
    table_visible=True,
    table_color=(0.34, 0.37, 0.38),
    table_roughness=0.38,
    table_metallic=0.42,
    dome_light_intensity=1650.0,
    dome_light_color=(0.90, 0.95, 1.0),
)

_KITCHEN_SINK = SceneProfile(
    name="kitchen_sink",
    description=(
        "Lightwheel Kitchen room aligned around the task-owned Sink054 asset "
        "without moving the shared robot or main camera."
    ),
    room_usd_path=(
        "forcebench://assets/lightwheel_kitchen/FullKitchen/"
        "KitchenRoom_background.usda"
    ),
    # Align the source KitchenRoom Sink054 root at
    # (2.277888, 1.333059, 0.849124) with the task sink root.
    room_translation_m=(-1.827888317449588, -1.0030593951112115, -0.89842848812593),
    room_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    hidden_room_prims=(),
    # The full Lightwheel room supplies the visual shell. The wrapper removes
    # all physics, its duplicate sink, and authored lights.
    visual_boxes=(),
    table_visual_usd_path=(
        "forcebench://assets/lightwheel_kitchen/Sink054/Sink054_environment.usda"
    ),
    # Sink054 is 0.874 m wide along native X and 0.538 m deep along native Y.
    # Rotate native width across camera-horizontal world Y. Its body top is
    # z=0.049304 m, so this translation aligns the counter exactly with z=0.
    table_visual_translation_m=(0.45, 0.33, -0.049304),
    table_visual_orientation_xyzw=(0.0, 0.0, -0.70710678, 0.70710678),
    table_visual_scale=(1.0, 1.0, 1.0),
    table_visual_collision_prim=None,
    table_size_m=(1.10, 1.50, 0.05),
    table_translation_m=(0.45, 0.33, -0.025),
    ground_center_z_m=-0.394,
    ground_visible=False,
    ground_color=(0.16, 0.17, 0.18),
    # Sink054_environment is the visible support; this remains the one stable
    # ForceBench collision plane until the basin proxy is introduced with the
    # interactive faucet articulation.
    table_visible=False,
    table_color=(0.52, 0.54, 0.53),
    table_roughness=0.36,
    table_metallic=0.0,
    dome_light_intensity=1700.0,
    dome_light_color=(0.94, 0.97, 1.0),
    evaluation_num_envs=10,
)

_KITCHEN_STOVE = SceneProfile(
    name="kitchen_stove",
    description=(
        "Lightwheel Kitchen room shared with faucet_turn and built around "
        "Stovetop012 without moving the shared robot or main camera."
    ),
    # Reuse the exact collision-free room shell and alignment from faucet_turn.
    # The stove keeps its own native support table and task-owned stovetop.
    room_usd_path=_KITCHEN_SINK.room_usd_path,
    room_translation_m=_KITCHEN_SINK.room_translation_m,
    room_orientation_xyzw=_KITCHEN_SINK.room_orientation_xyzw,
    hidden_room_prims=_KITCHEN_SINK.hidden_room_prims,
    visual_boxes=_KITCHEN_SINK.visual_boxes,
    table_visual_usd_path=None,
    table_visual_translation_m=(0.0, 0.0, 0.0),
    table_visual_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    table_visual_scale=(1.0, 1.0, 1.0),
    table_visual_collision_prim=None,
    table_size_m=(1.10, 1.50, 0.05),
    table_translation_m=(0.45, 0.10, -0.025),
    ground_center_z_m=-0.394,
    ground_visible=False,
    ground_color=(0.16, 0.17, 0.18),
    table_visible=True,
    table_color=(0.52, 0.54, 0.53),
    table_roughness=0.36,
    table_metallic=0.0,
    dome_light_intensity=1700.0,
    dome_light_color=(0.94, 0.97, 1.0),
    room_scale=_KITCHEN_SINK.room_scale,
    evaluation_num_envs=_KITCHEN_SINK.evaluation_num_envs,
)

_SIMPLE_ROOM_DRAWER = SceneProfile(
    name="simple_room_drawer",
    description=(
        "Bright low-clutter Simple Room-style laboratory for the sideways "
        "Sektion drawer task, preserving the canonical table and camera geometry."
    ),
    room_usd_path=None,
    room_translation_m=(0.0, 0.0, 0.0),
    room_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    hidden_room_prims=(),
    visual_boxes=(
        # A deliberately sparse, cool-toned room shell.  These boxes are visual
        # only; the exact-height ForceBench tabletop remains the sole support
        # collider, so changing the background cannot change task dynamics.
        VisualBox("simple_floor", (5.0, 5.0, 0.04), (0.48, 0.0, -0.8147), (0.56, 0.59, 0.61), roughness=0.86),
        VisualBox("simple_rear_wall", (0.08, 3.20, 2.40), (-1.58, 0.0, 0.43), (0.76, 0.79, 0.80), roughness=0.90),
        VisualBox("simple_left_wall", (4.20, 0.08, 2.40), (0.48, 1.48, 0.43), (0.69, 0.73, 0.75), roughness=0.90),
        VisualBox("simple_right_wall", (4.20, 0.08, 2.40), (0.48, -1.48, 0.43), (0.69, 0.73, 0.75), roughness=0.90),
        VisualBox("simple_wall_band", (0.035, 3.00, 0.22), (-1.535, 0.0, -0.40), (0.12, 0.36, 0.46), roughness=0.62),
        VisualBox("simple_window", (0.030, 1.10, 0.60), (-1.525, 0.0, 0.64), (0.18, 0.37, 0.45), roughness=0.22, metallic=0.12),
        VisualBox("simple_window_inner", (0.022, 0.94, 0.46), (-1.505, 0.0, 0.64), (0.56, 0.72, 0.78), roughness=0.18),
        VisualBox("simple_window_bar", (0.018, 0.035, 0.48), (-1.488, 0.0, 0.64), (0.18, 0.22, 0.24), roughness=0.34, metallic=0.55),
        VisualBox("simple_storage_left", (0.26, 0.46, 0.70), (-1.38, 0.98, -0.41), (0.30, 0.35, 0.37), roughness=0.44, metallic=0.36),
        VisualBox("simple_storage_right", (0.26, 0.46, 0.70), (-1.38, -0.98, -0.41), (0.30, 0.35, 0.37), roughness=0.44, metallic=0.36),
        VisualBox("simple_storage_top_left", (0.30, 0.50, 0.04), (-1.35, 0.98, -0.04), (0.73, 0.75, 0.74), roughness=0.55),
        VisualBox("simple_storage_top_right", (0.30, 0.50, 0.04), (-1.35, -0.98, -0.04), (0.73, 0.75, 0.74), roughness=0.55),
        VisualBox("simple_bench_front", (1.39, 0.045, 0.080), (0.475, -0.4225, -0.065), (0.17, 0.20, 0.22), roughness=0.36, metallic=0.62),
        VisualBox("simple_bench_back", (1.39, 0.045, 0.080), (0.475, 0.4225, -0.065), (0.17, 0.20, 0.22), roughness=0.36, metallic=0.62),
        VisualBox("simple_bench_leg_fl", (0.055, 0.055, 0.730), (-0.165, -0.365, -0.415), (0.17, 0.20, 0.22), roughness=0.36, metallic=0.62),
        VisualBox("simple_bench_leg_fr", (0.055, 0.055, 0.730), (1.115, -0.365, -0.415), (0.17, 0.20, 0.22), roughness=0.36, metallic=0.62),
        VisualBox("simple_bench_leg_bl", (0.055, 0.055, 0.730), (-0.165, 0.365, -0.415), (0.17, 0.20, 0.22), roughness=0.36, metallic=0.62),
        VisualBox("simple_bench_leg_br", (0.055, 0.055, 0.730), (1.115, 0.365, -0.415), (0.17, 0.20, 0.22), roughness=0.36, metallic=0.62),
        VisualBox("simple_ceiling_light_l", (1.15, 0.12, 0.035), (-0.20, -0.68, 1.40), (0.84, 0.87, 0.88), roughness=0.24, emissive_color=(4.5, 4.8, 5.0)),
        VisualBox("simple_ceiling_light_r", (1.15, 0.12, 0.035), (-0.20, 0.68, 1.40), (0.84, 0.87, 0.88), roughness=0.24, emissive_color=(4.5, 4.8, 5.0)),
    ),
    table_visual_usd_path=None,
    table_visual_translation_m=(0.0, 0.0, 0.0),
    table_visual_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    table_visual_scale=(1.0, 1.0, 1.0),
    table_visual_collision_prim=None,
    table_size_m=(1.35, 0.80, 0.05),
    table_translation_m=(0.475, 0.0, -0.025),
    ground_center_z_m=-0.8447,
    ground_visible=False,
    ground_color=(0.56, 0.59, 0.61),
    table_visible=True,
    table_color=(0.64, 0.66, 0.65),
    table_roughness=0.42,
    table_metallic=0.08,
    dome_light_intensity=1850.0,
    dome_light_color=(0.92, 0.97, 1.0),
)

_SIMPLE_ROOM_SORT_TABLE = SceneProfile(
    name="simple_room_sort_table",
    description=(
        "The bright Simple Room-style drawer background with the exact EastRural "
        "table used by the box weight-sorting task."
    ),
    room_usd_path=None,
    room_translation_m=(0.0, 0.0, 0.0),
    room_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    hidden_room_prims=(),
    # EastRural supplies the visible table, so remove the synthetic bench frame
    # while retaining the same clean room shell and background storage.
    visual_boxes=tuple(
        box
        for box in _SIMPLE_ROOM_DRAWER.visual_boxes
        if not box.name.startswith("simple_bench_")
    ),
    table_visual_usd_path=(
        "forcebench://assets/residential/east_rural/EastRural_Table_weight_sort.usda"
    ),
    table_visual_translation_m=(0.40, 0.0, -0.72),
    table_visual_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    table_visual_scale=(0.01202975, 0.01014864, 0.00944882),
    table_visual_collision_prim="collisions",
    table_size_m=(1.10, 1.55, 0.05),
    table_translation_m=(0.40, 0.0, -0.025),
    ground_center_z_m=-0.77,
    ground_visible=False,
    ground_color=(0.56, 0.59, 0.61),
    # Keep one exact-height invisible ForceBench collider.  The EastRural USD is
    # the box-sort table appearance and its collision branch is disabled.
    table_visible=False,
    table_color=(0.42, 0.30, 0.19),
    table_roughness=0.64,
    table_metallic=0.0,
    dome_light_intensity=1850.0,
    dome_light_color=(0.92, 0.97, 1.0),
)

_WHITEBOARD_SIMPLE_ROOM = SceneProfile(
    name="whiteboard_simple_room",
    description=(
        "Official Isaac Sim Simple Room used as a collision-free visual shell "
        "around the ForceBench tabletop chalkboard workspace."
    ),
    room_usd_path="Environments/Simple_Room/simple_room.usd",
    # The official room floor is authored at z=0.  Lower it to the same floor
    # elevation as the ForceBench table legs while retaining its walls, wood
    # accents and practical indoor lighting context.
    room_translation_m=(0.0, 0.0, -0.7947),
    room_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    hidden_room_prims=(),
    visual_boxes=(),
    table_visual_usd_path=None,
    table_visual_translation_m=(0.0, 0.0, 0.0),
    table_visual_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
    table_visual_scale=(1.0, 1.0, 1.0),
    table_visual_collision_prim=None,
    table_size_m=(1.35, 0.80, 0.05),
    table_translation_m=(0.475, 0.0, -0.025),
    ground_center_z_m=-0.8447,
    ground_visible=False,
    ground_color=(0.60, 0.58, 0.54),
    table_visible=True,
    table_color=(0.60, 0.62, 0.61),
    table_roughness=0.43,
    table_metallic=0.07,
    dome_light_intensity=1550.0,
    dome_light_color=(0.96, 0.96, 0.94),
)

# Lightweight classroom alternatives built from a small subset of the
# official Isaac Sim Office asset.  Keeping each composition in a local USDA
# wrapper avoids instantiating the complete 4k-prim office in every cloned
# environment while preserving the official authored meshes and materials.
_WHITEBOARD_CLASSROOM = replace(
    _WHITEBOARD_SIMPLE_ROOM,
    name="whiteboard_classroom",
    description=(
        "Lightweight classroom background composed from official Isaac Sim "
        "Office wall, bookcase and clock assets."
    ),
    room_usd_path=(
        "forcebench://assets/wipe_board_classroom/classroom_background.usda"
    ),
    room_translation_m=(0.0, 0.0, 0.0),
    dome_light_intensity=1450.0,
    dome_light_color=(0.98, 0.97, 0.93),
)

_WHITEBOARD_CLASSROOM_WALL = replace(
    _WHITEBOARD_CLASSROOM,
    name="whiteboard_classroom_wall",
    description="Diagnostic classroom render containing only the official Office wall.",
    room_usd_path="forcebench://assets/wipe_board_classroom/wall_only.usda",
)

_WHITEBOARD_CLASSROOM_BOOKCASE = replace(
    _WHITEBOARD_CLASSROOM,
    name="whiteboard_classroom_bookcase",
    description="Diagnostic classroom render containing only the official Office bookcase.",
    room_usd_path="forcebench://assets/wipe_board_classroom/bookcase_only.usda",
)

_WHITEBOARD_CLASSROOM_CLOCK = replace(
    _WHITEBOARD_CLASSROOM,
    name="whiteboard_classroom_clock",
    description="Diagnostic classroom render containing only the official Office clock.",
    room_usd_path="forcebench://assets/wipe_board_classroom/clock_only.usda",
)

_SCENE_PROFILES = {
    profile.name: profile
    for profile in (
        _CONTACT_LAB,
        _WEIGHT_SORT_RESIDENTIAL,
        _PRECISION_ASH_VENEER,
        _PARCEL_SORT_STATION,
        _PARCEL_SORT_STATION_COLLECTION_20260823,
        _GEAR_ASSEMBLY_CELL,
        _KITCHEN_SINK,
        _KITCHEN_STOVE,
        _SIMPLE_ROOM_DRAWER,
        _SIMPLE_ROOM_SORT_TABLE,
        _WHITEBOARD_SIMPLE_ROOM,
        _WHITEBOARD_CLASSROOM,
        _WHITEBOARD_CLASSROOM_WALL,
        _WHITEBOARD_CLASSROOM_BOOKCASE,
        _WHITEBOARD_CLASSROOM_CLOCK,
    )
}
for _profile in _SCENE_PROFILES.values():
    _profile.validate()


def scene_profile_names() -> tuple[str, ...]:
    """Return stable profile names for command-line choices and metadata."""

    return tuple(_SCENE_PROFILES)


def get_scene_profile(name: str) -> SceneProfile:
    """Resolve a profile by name with an actionable error."""

    try:
        return _SCENE_PROFILES[str(name)]
    except KeyError as exc:
        choices = ", ".join(scene_profile_names())
        raise ValueError(f"unknown scene profile {name!r}; choose one of: {choices}") from exc
