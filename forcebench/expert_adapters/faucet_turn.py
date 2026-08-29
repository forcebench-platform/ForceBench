"""Disabled per-seed adapter for the canonical 32-clone faucet collector."""

from forcebench.expert_api import ExpertBackend, ExpertBackendManifest, ExpertRequest

MANIFEST = ExpertBackendManifest(
    name="scripted",
    task="faucet_turn",
    backend_class="forcebench.expert_adapters.faucet_turn:FaucetTurnExpertBackend",
    description="Retired per-seed adapter; use the canonical 32-clone collector.",
    enabled=False,
    disabled_reason="faucet_turn is collected only through the 32-clone batch entry point",
)


class FaucetTurnExpertBackend(ExpertBackend):
    manifest = MANIFEST

    def build_command(self, request: ExpertRequest) -> list[str]:
        raise RuntimeError(
            "the per-seed faucet_turn backend is retired; run "
            "run_native.sh expert --task faucet_turn to use 32 cloned workspaces"
        )
