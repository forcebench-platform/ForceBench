"""Scripted expert plugin for ``weight_sort``."""

from coribench.expert_api import ExpertBackend, ExpertBackendManifest, ExpertRequest


MANIFEST = ExpertBackendManifest(
    name="scripted",
    task="weight_sort",
    backend_class="coribench.expert_adapters.weight_sort:WeightSortExpertBackend",
    description="Retired per-seed adapter; use the default 32-clone collector.",
    enabled=False,
    disabled_reason="weight_sort is collected only through the 32-clone batch entry point",
)


class WeightSortExpertBackend(ExpertBackend):
    manifest = MANIFEST

    def build_command(self, request: ExpertRequest) -> list[str]:
        raise RuntimeError(
            "the per-seed weight_sort backend is retired; run "
            "run_native.sh expert --task weight_sort to use 32 cloned workspaces"
        )
