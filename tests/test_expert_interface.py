import json
import os
from pathlib import Path
import signal
import sys

import numpy as np
import pytest

import coribench.expert_collection as expert_collection
from coribench.expert_api import EXPERT_COLLECTION_CONTRACT, ExpertRequest
from coribench.expert_collection import (
    EMA_WRENCH_FIELD,
    RAW_WRENCH_FIELD,
    RECORDED_WRENCH_FIELDS,
    parse_seed_spec,
)
from coribench.expert_registry import (
    all_expert_backend_manifests,
    default_expert_backend_name,
    expert_backend_manifests,
    expert_backend_names,
    load_expert_backend,
)
from coribench.recording import EpisodeBuffer, HDF5EpisodeWriter
from scripts.expert.parallel_sort_collection import (
    DEFAULT_CLONE_COUNT,
    PARALLEL_TASKS,
    build_clone_batches,
)


SCRIPT_PATHS = (
    "scripts/expert/run_persistent_expert_worker.py",
    "scripts/expert/generate_faucet_peak_expert.py",
    "scripts/expert/generate_scripted_gear_mesh.py",
    "scripts/expert/generate_privileged_nut_thread.py",
    "scripts/expert/generate_parallel_peg_insert.py",
    "scripts/expert/generate_parallel_radmanso.py",
    "scripts/expert/generate_parallel_sort.py",
    "scripts/expert/generate_parallel_stove.py",
    "scripts/expert/parallel_sort_collection.py",
)


def fake_roots(tmp_path: Path) -> tuple[Path, Path]:
    code_root = tmp_path / "code"
    isaaclab_root = tmp_path / "isaaclab"
    for relative in SCRIPT_PATHS:
        path = code_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test\n", encoding="utf-8")
    isaaclab_root.mkdir()
    (isaaclab_root / "isaaclab.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    return code_root.resolve(), isaaclab_root.resolve()


def request_for(
    tmp_path: Path,
    task: str,
    backend: str,
    *,
    checkpoint: Path | None = None,
    record_video: bool = False,
    options=None,
) -> ExpertRequest:
    code_root, isaaclab_root = fake_roots(tmp_path)
    return ExpertRequest(
        task=task,
        backend=backend,
        seed=7,
        attempt=1,
        code_root=code_root,
        isaaclab_root=isaaclab_root,
        dataset_dir=(tmp_path / "formal" / task).resolve(),
        artifact_dir=(tmp_path / "scratch" / task).resolve(),
        record_video=record_video,
        checkpoint=checkpoint,
        options=options or {},
    )


def test_expert_backends_are_registered_per_task():
    manifests = expert_backend_manifests()
    assert manifests
    assert all(key == (manifest.task, manifest.name) for key, manifest in manifests.items())
    assert expert_backend_names("radmanso_drawer_sort_table") == ()
    assert expert_backend_names("peg_insert") == ()
    assert default_expert_backend_name("gear_mesh") == "scripted"
    assert default_expert_backend_name("nut_thread") == "scripted"
    assert expert_backend_names("weight_sort") == ()
    assert expert_backend_names("rattle_sort") == ()
    assert expert_backend_names("stove_turn_on") == ()
    all_manifests = all_expert_backend_manifests()
    assert not all_manifests[("weight_sort", "scripted")].enabled
    assert not all_manifests[("rattle_sort", "scripted")].enabled
    assert not all_manifests[("stove_turn_on", "scripted")].enabled
    assert not all_manifests[("radmanso_drawer_sort_table", "scripted")].enabled
    assert not all_manifests[("peg_insert", "scripted")].enabled
    with pytest.raises(ValueError, match="no automated expert backend"):
        default_expert_backend_name("radmanso_drawer_sort_table")
    with pytest.raises(ValueError, match="no automated expert backend"):
        default_expert_backend_name("peg_insert")


def test_every_scripted_task_builds_one_isolated_seed_command(tmp_path):
    scripted_tasks = (
        "faucet_turn",
        "gear_mesh",
        "nut_thread",
    )
    for index, task in enumerate(scripted_tasks):
        request = request_for(tmp_path / str(index), task, "scripted")
        command = load_expert_backend(task, "scripted").build_command(request)
        assert command[:2] == [str(request.isaaclab_root / "isaaclab.sh"), "-p"]
        assert str(request.dataset_dir) in command
        assert str(request.seed) in command
        assert "--enable_cameras" in command


def test_faucet_expert_uses_weight_sort_physics_rate(tmp_path):
    request = request_for(tmp_path, "faucet_turn", "scripted")
    command = load_expert_backend("faucet_turn", "scripted").build_command(request)
    rate_index = command.index("--physics_rate_hz")
    assert command[rate_index + 1] == "60"
    solver_index = command.index("--solver_position_iterations")
    assert command[solver_index + 1] == "64"


def test_faucet_expert_transition_uses_only_physical_valve_state():
    source = Path("scripts/expert/generate_faucet_peak_expert.py").read_text(
        encoding="utf-8"
    )
    assert "force_sensor_task_compensated_smooth" not in source
    assert "wrist_torque_z_nm" not in source
    assert '"expert_transition_signal": "valve_joint_angle_rad"' in source
    assert '"physical_transition_condition"' in source
    assert "wrench_raw=vector(env.force_sensor_joint_raw[0])" in source
    assert "wrench_ema=vector(env.force_sensor_joint_smooth[0])" in source


def test_sort_batches_always_use_32_distinct_clones_and_skip_existing():
    assert "stove_turn_on" in PARALLEL_TASKS
    assert "radmanso_drawer_sort_table" in PARALLEL_TASKS
    assert "peg_insert" in PARALLEL_TASKS
    requested = tuple(range(70))
    existing = set(range(10))
    batches = build_clone_batches(requested, existing)
    assert [len(batch.seeds) for batch in batches] == [DEFAULT_CLONE_COUNT, DEFAULT_CLONE_COUNT]
    assert all(len(set(batch.seeds)) == DEFAULT_CLONE_COUNT for batch in batches)
    assert tuple(seed for batch in batches for seed in batch.write_seeds) == tuple(range(10, 70))
    assert not existing.intersection(
        seed for batch in batches for seed in batch.write_seeds
    )

def test_assembly_backends_are_task_specific_scripts(tmp_path):
    expected_scripts = {
        "gear_mesh": "generate_scripted_gear_mesh.py",
        "nut_thread": "generate_privileged_nut_thread.py",
    }
    for index, (task, script) in enumerate(expected_scripts.items()):
        request = request_for(tmp_path / str(index), task, "scripted")
        command = load_expert_backend(task, "scripted").build_command(request)
        assert command[2].endswith(script)
        assert "--checkpoint" not in command
        if task == "gear_mesh":
            assert command[command.index("--variant") + 1] == "direct"
        else:
            assert "--variant" not in command


def test_seed_spec_is_unambiguous_and_deduplicated():
    assert parse_seed_spec("0,2-4,8:13:2,4") == (0, 2, 3, 4, 8, 10, 12)
    with pytest.raises(ValueError):
        parse_seed_spec("4-2")
    with pytest.raises(ValueError):
        parse_seed_spec("1,,2")


def test_expert_contract_records_raw_and_ema_but_uses_raw_for_policy():
    assert EXPERT_COLLECTION_CONTRACT == "coribench_expert_collection"
    assert RAW_WRENCH_FIELD.endswith("eef_wrench_joint_raw")
    assert EMA_WRENCH_FIELD.endswith("eef_wrench_joint_ema")
    assert RECORDED_WRENCH_FIELDS == (RAW_WRENCH_FIELD, EMA_WRENCH_FIELD)


def test_training_and_inference_use_raw_wrench():
    root = Path(__file__).resolve().parents[1]
    converter = (root / "convert_hdf5_to_lerobot.py").read_text(encoding="utf-8")
    # Renamed from inference_coribench.py during the 2026-08-25 vectorized-eval
    # migration; the old name is archived, not part of this extraction.
    inference = (
        root / "scripts/evaluation/inference_coribench_vectorized.py"
    ).read_text(encoding="utf-8")
    assert 'handle["forces/control/eef_wrench_joint_raw"]' in converter
    # The vectorized eval loop batches this across all cloned envs
    # (`env.force_sensor_joint_raw`, indexed per-env later) rather than
    # reading a hardcoded [0]; the invariant under test is "raw, not EMA".
    assert "env.force_sensor_joint_raw" in inference
    assert "env.force_sensor_joint_ema" not in inference
    assert "env.force_sensor_joint_smooth[0]" not in inference


def test_standard_supervisor_validates_and_resumes_by_seed(tmp_path, monkeypatch):
    code_root, isaaclab_root = fake_roots(tmp_path)
    data_root = (tmp_path / "data").resolve()
    scratch_root = (tmp_path / "scratch").resolve()
    monkeypatch.setattr(expert_collection, "DATA_ROOT", data_root)
    monkeypatch.setattr(expert_collection, "SCRATCH_ROOT", scratch_root)
    dataset = data_root / "datasets" / "faucet_turn"
    run_dir = scratch_root / "expert" / "faucet_turn" / "test_run"

    def fake_run(command, *, cwd, environment, log_path, timeout_s):
        seed = int(command[command.index("--seed") + 1])
        episode = EpisodeBuffer()
        episode.append(
            state=np.zeros(7),
            action=np.zeros(7),
            wrench_raw=np.arange(6, dtype=np.float32),
            wrench_ema=np.arange(6, dtype=np.float32) / 2,
            timestamp=1 / 15,
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            wrist_image=np.zeros((8, 8, 3), dtype=np.uint8),
            joint_position=np.zeros(9),
            joint_velocity=np.zeros(9),
            tcp_pose=np.zeros(7),
            privileged={"objects/box/pose": np.zeros(7)},
        )
        HDF5EpisodeWriter(dataset).write(
            episode,
            {
                "task": "faucet_turn",
                "success": True,
                "seed": seed,
                "initial_layout": {},
                "scene_profile": "kitchen_sink",
                "control_source": "test_scripted",
                "images_present": True,
            },
            high_rate={
                "timestamp": np.asarray([1 / 60], dtype=np.float64),
                "eef_wrench_joint_raw": np.zeros((1, 6), dtype=np.float32),
            },
        )
        return 0, False, 0.01

    monkeypatch.setattr(expert_collection, "_run_process", fake_run)
    plan = expert_collection.ExpertCollectionPlan(
        task="faucet_turn",
        backend="scripted",
        seeds=(5,),
        dataset_dir=dataset,
        run_dir=run_dir,
        isaaclab_root=isaaclab_root,
        code_root=code_root,
        launch_stagger_s=0,
        persistent_workers=False,
    )
    summary = expert_collection.execute_expert_collection(plan)
    assert summary["successful_seeds"] == [5]
    assert summary["valid_episode_counts"] == {"5": 1}
    assert (dataset / "seed_005_episode_01.hdf5").is_file()

    resumed = expert_collection.execute_expert_collection(plan)
    assert resumed["skipped_complete_seeds"] == [5]
    assert resumed["results"] == []


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup is POSIX-specific")
@pytest.mark.parametrize("leader_exits", (False, True))
def test_supervisor_reaps_term_resistant_isaac_descendants(
    tmp_path,
    monkeypatch,
    leader_exits,
):
    process_info = tmp_path / "process_info.txt"
    log_path = tmp_path / "attempt.log"
    script = """
import os
from pathlib import Path
import signal
import sys
import time

child = os.fork()
if child == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(sys.argv[1]).write_text(f"{os.getpid()} {os.getpgrp()}", encoding="utf-8")
    while True:
        time.sleep(1.0)
if sys.argv[2] == "exit":
    raise SystemExit(0)
while True:
    time.sleep(1.0)
"""
    monkeypatch.setattr(expert_collection, "_PROCESS_TERMINATE_GRACE_S", 0.2)
    monkeypatch.setattr(expert_collection, "_PROCESS_KILL_GRACE_S", 1.0)
    monkeypatch.setattr(expert_collection, "_PROCESS_GROUP_POLL_S", 0.01)
    command = [
        sys.executable,
        "-c",
        script,
        str(process_info),
        "exit" if leader_exits else "wait",
    ]
    child_pid = None
    process_group_id = None
    try:
        return_code, timed_out, _ = expert_collection._run_process(
            command,
            cwd=tmp_path,
            environment=os.environ.copy(),
            log_path=log_path,
            timeout_s=0.5,
        )
        child_pid, process_group_id = (
            int(value) for value in process_info.read_text(encoding="utf-8").split()
        )
        assert timed_out is (not leader_exits)
        assert return_code == (0 if leader_exits else -signal.SIGTERM)
        assert not expert_collection._process_group_exists(process_group_id)
        log = log_path.read_text(encoding="utf-8")
        assert "sending SIGKILL" in log
    finally:
        if process_group_id and expert_collection._process_group_exists(process_group_id):
            os.killpg(process_group_id, signal.SIGKILL)
        elif child_pid:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
