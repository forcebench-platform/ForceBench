"""Physical scripted expert plugin for ``gear_mesh``."""

from forcebench.expert_api import ExpertBackend, ExpertBackendManifest, ExpertRequest

from ._command import (
    append_options,
    canonical_physics_args,
    common_camera_args,
    isaac_command,
    video_path,
)


MANIFEST = ExpertBackendManifest(
    name="scripted",
    task="gear_mesh",
    backend_class="forcebench.expert_adapters.gear_mesh:GearMeshExpertBackend",
    description="Closed-loop gear centering, tooth search, seating, release, and validation.",
)


class GearMeshExpertBackend(ExpertBackend):
    manifest = MANIFEST

    def build_command(self, request: ExpertRequest) -> list[str]:
        request.validate()
        if request.task != self.manifest.task:
            raise ValueError(f"this backend belongs to {self.manifest.task!r}")
        command = [
            *isaac_command(request, "scripts/expert/generate_scripted_gear_mesh.py"),
            "--output_dir",
            str(request.dataset_dir),
            "--seed",
            str(request.seed),
            "--variant",
            "direct",
            *common_camera_args(request),
            *canonical_physics_args(request),
        ]
        if request.record_video:
            command.extend(("--video", str(video_path(request))))
        append_options(
            command,
            request.options,
            {
                "offset_radius_min_m": (int, float),
                "offset_radius_max_m": (int, float),
            },
        )
        return command
