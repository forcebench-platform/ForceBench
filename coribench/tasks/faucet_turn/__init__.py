"""Faucet-turn environment manifest (safe to import before Isaac Sim starts)."""

from coribench.task_api import TaskManifest


MANIFEST = TaskManifest(
    name="faucet_turn",
    prompt="Rotate the faucet knob past the detent to turn on the water.",
    runtime_class="coribench.tasks.faucet_turn.runtime:FaucetTurnTask",
    dataset_subdir="faucet_turn",
    scene_profile="kitchen_sink",
    description="Kitchen faucet manipulation with a nonlinear snap-through detent.",
)
