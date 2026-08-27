import ast
from pathlib import Path

from forcebench.registry import get_task_manifest, task_names


def test_task_plugins_are_auto_discovered_without_simulator_imports():
    discovered = set(task_names())
    assert {
        "faucet_turn",
        "gear_mesh",
        "nut_thread",
        "peg_insert",
        "radmanso_drawer_sort_table",
        "rattle_sort",
        "stove_turn_on",
        "weight_sort",
    } <= discovered
    assert not {
        "friction_sort",
        "press_turn_cap",
        "spring_stiffness_qc",
        "seatbelt_buckle_2x",
        "drawer_latch_stop",
        "tension_stop_probe",
        "sektion_drawer_sort_table",
        "sideways_sektion_drawer",
        "seatbelt_buckle",
        "tension_stop_drawer",
    } & discovered
    manifest = get_task_manifest("peg_insert")
    assert manifest.runtime_class.endswith(":PegInsertTask")
    assert manifest.dataset_subdir == "peg_insert"
    # peg_insert and nut_thread moved off gear_assembly_cell to
    # precision_ash_veneer at some point after this test was first written;
    # gear_mesh is the only one still on gear_assembly_cell (see the other
    # assertion below). Verified against the live manifests, not assumed.
    assert manifest.scene_profile == "precision_ash_veneer"
    assert manifest.prompt == (
        "Pick up the silver rod from the blue fixture, insert it into the bronze "
        "target hole, release it, and retract the gripper."
    )


def test_task_manifests_own_scene_selection_and_dataset_names():
    manifests = [get_task_manifest(name) for name in task_names()]
    assert len({item.dataset_subdir for item in manifests}) == len(manifests)
    assert get_task_manifest("gear_mesh").scene_profile == "gear_assembly_cell"
    assert {
        get_task_manifest(name).scene_profile
        for name in ("nut_thread", "peg_insert")
    } == {"precision_ash_veneer"}
    assert get_task_manifest("weight_sort").scene_profile == "weight_sort_residential"
    assert get_task_manifest("rattle_sort").scene_profile == "parcel_sort_station"
    assert get_task_manifest("radmanso_drawer_sort_table").scene_profile == (
        "simple_room_sort_table"
    )


def test_task_manifests_own_formal_evaluation_step_limits():
    assert {
        get_task_manifest(name).evaluation_max_steps
        for name in ("gear_mesh", "nut_thread", "peg_insert")
    } == {1500}
    assert {
        get_task_manifest(name).evaluation_max_steps
        for name in task_names()
        if name not in {"gear_mesh", "nut_thread", "peg_insert"}
    } == {600}


def test_core_runtime_has_no_factory_or_peg_plugin_imports():
    root = Path(__file__).resolve().parents[1]
    # scripts/collection/record_forcebench.py (GELLO real-hardware teleop) is
    # out of scope for ForceBench (sim-only) and is not part of this
    # extraction; the factory/peg-import invariant is checked on the shared
    # sim runtime files only.
    core_files = [
        root / "forcebench" / "env.py",
        root / "forcebench" / "env_cfg.py",
    ]
    for path in core_files:
        source = path.read_text(encoding="utf-8")
        assert "isaaclab_tasks.direct.factory" not in source
        assert "forcebench.tasks.peg_insert" not in source
        assert "PEG_INSERT" not in source


def test_core_runtime_invokes_the_public_control_step_hook():
    root = Path(__file__).resolve().parents[1]
    source = (root / "forcebench" / "env.py").read_text(encoding="utf-8")
    assert "self.task.on_control_step(self)" in source


def test_inference_loop_uses_only_the_policy_adapter_contract():
    root = Path(__file__).resolve().parents[1]
    # Renamed from inference_forcebench.py during the 2026-08-25 vectorized-eval
    # migration; the old name is archived, not part of this extraction.
    source = (
        root / "scripts" / "evaluation" / "inference_forcebench_vectorized.py"
    ).read_text(encoding="utf-8")
    assert "load_policy_adapter" in source
    assert "PolicyObservation" in source
    assert "observation/state" not in source
    assert "OpenPIWebSocketClient" not in source
    assert "policy_input_schema" not in source


def test_first_party_launchers_only_reference_registered_arm_modes():
    root = Path(__file__).resolve().parents[1]
    retired_modes = (
        "_".join(("hard", "position")),
        "_".join(("dynamic", "position")),
    )
    for path in (*root.rglob("*.py"), *root.rglob("*.sh")):
        relative = path.relative_to(root)
        if relative.parts[0] in {"assets", "runtime"}:
            continue
        source = path.read_text(encoding="utf-8")
        assert not any(mode in source for mode in retired_modes), path


def test_all_tasks_use_one_global_start_joint_position():
    root = Path(__file__).resolve().parents[1]
    env_cfg_path = root / "forcebench" / "env_cfg.py"
    source = env_cfg_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "START_JOINT_POSITIONS"
    )
    assert ast.literal_eval(assignment.value) == (
        0.0086831,
        0.0687144,
        -0.0087775,
        -2.3834085,
        0.0004833,
        2.4512002,
        0.7849010,
    )
    for path in (root / "forcebench").rglob("*.py"):
        task_source = path.read_text(encoding="utf-8")
        assert "STANDBY_JOINT_POSITIONS" not in task_source
        assert "standby_joint_positions" not in task_source
