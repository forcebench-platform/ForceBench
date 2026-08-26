"""Scripted expert plugin for ``rattle_sort``."""

from coribench.expert_api import ExpertBackend, ExpertBackendManifest, ExpertRequest


MANIFEST = ExpertBackendManifest(
    name="scripted",
    task="rattle_sort",
    backend_class="coribench.expert_adapters.rattle_sort:RattleSortExpertBackend",
    description="Retired per-seed adapter; use the default 32-clone collector.",
    enabled=False,
    disabled_reason="rattle_sort is collected only through the 32-clone batch entry point",
)


class RattleSortExpertBackend(ExpertBackend):
    manifest = MANIFEST

    def build_command(self, request: ExpertRequest) -> list[str]:
        raise RuntimeError(
            "the per-seed rattle_sort backend is retired; run "
            "run_native.sh expert --task rattle_sort to use 32 cloned workspaces"
        )
