"""Scripted expert plugin for ``stove_turn_on``."""

from coribench.expert_api import ExpertBackend, ExpertBackendManifest, ExpertRequest

MANIFEST = ExpertBackendManifest(
    name="scripted",
    task="stove_turn_on",
    backend_class="coribench.expert_adapters.stove_turn_on:StoveTurnOnExpertBackend",
    description="Retired per-seed adapter; use the default 32-clone collector.",
    enabled=False,
    disabled_reason="stove_turn_on is collected only through the 32-clone batch entry point",
)


class StoveTurnOnExpertBackend(ExpertBackend):
    manifest = MANIFEST

    def build_command(self, request: ExpertRequest) -> list[str]:
        raise RuntimeError(
            "the per-seed stove_turn_on backend is retired; run "
            "run_native.sh expert --task stove_turn_on to use 32 cloned workspaces"
        )
