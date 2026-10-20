"""Disabled per-seed adapter for the canonical 32-clone nut collector."""

from forcebench.expert_api import ExpertBackend, ExpertBackendManifest, ExpertRequest

MANIFEST = ExpertBackendManifest(
    name="scripted",
    task="nut_thread",
    backend_class="forcebench.expert_adapters.nut_thread:NutThreadExpertBackend",
    description="Retired per-seed adapter; use the canonical 32-clone collector.",
    enabled=False,
    disabled_reason="nut_thread is collected only through the 32-clone batch entry point",
)


class NutThreadExpertBackend(ExpertBackend):
    manifest = MANIFEST

    def build_command(self, request: ExpertRequest) -> list[str]:
        raise RuntimeError(
            "the per-seed nut_thread backend is retired; run "
            "run_native.sh expert --task nut_thread to use 32 cloned workspaces"
        )
