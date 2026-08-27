"""Scripted expert plugin for ``faucet_turn``."""

from forcebench.expert_api import ExpertBackend, ExpertBackendManifest, ExpertRequest

from ._command import append_options, isaac_command


FAUCET_PHYSICS_RATE_HZ = 60


MANIFEST = ExpertBackendManifest(
    name="scripted",
    task="faucet_turn",
    backend_class="forcebench.expert_adapters.faucet_turn:FaucetTurnExpertBackend",
    description="Detect the faucet detent, back off, release, and validate success.",
)


class FaucetTurnExpertBackend(ExpertBackend):
    manifest = MANIFEST

    def build_command(self, request: ExpertRequest) -> list[str]:
        request.validate()
        if request.task != self.manifest.task:
            raise ValueError(f"this backend belongs to {self.manifest.task!r}")
        command = [
            *isaac_command(request, "scripts/expert/generate_faucet_peak_expert.py"),
            "--seed",
            str(request.seed),
            "--output_dir",
            str(request.artifact_dir),
            "--dataset_dir",
            str(request.dataset_dir),
            "--physics_rate_hz",
            str(FAUCET_PHYSICS_RATE_HZ),
            "--solver_position_iterations",
            str(request.solver_position_iterations),
            "--headless",
            "--enable_cameras",
            "--device",
            request.device,
        ]
        append_options(
            command,
            request.options,
            {
                "forearm_effort_limit": (int, float),
                "gripper_contact_friction": (int, float),
                "hold_frames": int,
                "max_refine_frames": int,
                "motion_profile": str,
            },
        )
        return command
