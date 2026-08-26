"""Privileged closed-loop expert plugin for ``nut_thread``."""

from coribench.expert_api import ExpertBackend, ExpertBackendManifest, ExpertRequest

from ._command import append_options, canonical_physics_args, common_camera_args, isaac_command


MANIFEST = ExpertBackendManifest(
    name="scripted",
    task="nut_thread",
    backend_class="coribench.expert_adapters.nut_thread:NutThreadExpertBackend",
    description="Exact nut/bolt-pose teacher through SmoothPosition and physical contact.",
)


class NutThreadExpertBackend(ExpertBackend):
    manifest = MANIFEST

    def build_command(self, request: ExpertRequest) -> list[str]:
        request.validate()
        if request.task != self.manifest.task:
            raise ValueError(f"this backend belongs to {self.manifest.task!r}")
        command = [
            *isaac_command(request, "scripts/expert/generate_privileged_nut_thread.py"),
            "--output_dir", str(request.dataset_dir),
            "--artifact_dir", str(request.artifact_dir),
            "--episodes", "1",
            "--start_seed", str(request.seed),
            "--max_attempts_per_episode", "1",
            *common_camera_args(request),
            *canonical_physics_args(request),
        ]
        append_options(command, request.options, {"translation_speed_m_s": (int, float)})
        return command
