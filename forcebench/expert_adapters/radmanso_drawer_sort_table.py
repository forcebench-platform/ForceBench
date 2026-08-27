"""Disabled per-seed adapter for the default 32-clone RADMANSO collector."""

from forcebench.expert_api import ExpertBackend, ExpertBackendManifest, ExpertRequest


MANIFEST = ExpertBackendManifest(
    name="scripted",
    task="radmanso_drawer_sort_table",
    backend_class=(
        "forcebench.expert_adapters.radmanso_drawer_sort_table:"
        "RadmansoDrawerSortTableExpertBackend"
    ),
    description="Retired per-seed adapter; use the default 32-clone collector.",
    enabled=False,
    disabled_reason=(
        "radmanso_drawer_sort_table is collected only through the 32-clone "
        "batch entry point"
    ),
)


class RadmansoDrawerSortTableExpertBackend(ExpertBackend):
    manifest = MANIFEST

    def build_command(self, request: ExpertRequest) -> list[str]:
        raise RuntimeError(
            "the per-seed RADMANSO backend is retired; run "
            "run_native.sh expert --task radmanso_drawer_sort_table to use "
            "32 cloned workspaces"
        )
