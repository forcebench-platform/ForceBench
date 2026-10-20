"""Scripted expert plugin for ``peg_insert``."""

from forcebench.expert_api import ExpertBackend, ExpertBackendManifest, ExpertRequest


MANIFEST = ExpertBackendManifest(
    name="scripted",
    task="peg_insert",
    backend_class="forcebench.expert_adapters.peg_insert:PegInsertExpertBackend",
    description="Retired per-seed adapter; use the default 32-clone collector.",
    enabled=False,
    disabled_reason="peg_insert is collected only through the 32-clone batch entry point",
)


class PegInsertExpertBackend(ExpertBackend):
    manifest = MANIFEST

    def build_command(self, request: ExpertRequest) -> list[str]:
        raise RuntimeError(
            "the per-seed peg_insert backend is retired; run "
            "run_native.sh expert --task peg_insert to use 32 cloned workspaces"
        )
