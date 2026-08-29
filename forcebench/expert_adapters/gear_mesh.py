"""Disabled per-seed adapter for the canonical 32-clone gear collector."""

from forcebench.expert_api import ExpertBackend, ExpertBackendManifest, ExpertRequest

MANIFEST = ExpertBackendManifest(
    name="scripted",
    task="gear_mesh",
    backend_class="forcebench.expert_adapters.gear_mesh:GearMeshExpertBackend",
    description="Retired per-seed adapter; use the canonical 32-clone collector.",
    enabled=False,
    disabled_reason="gear_mesh is collected only through the 32-clone batch entry point",
)


class GearMeshExpertBackend(ExpertBackend):
    manifest = MANIFEST

    def build_command(self, request: ExpertRequest) -> list[str]:
        raise RuntimeError(
            "the per-seed gear_mesh backend is retired; run "
            "run_native.sh expert --task gear_mesh to use 32 cloned workspaces"
        )
