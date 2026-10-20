"""CPU routing checks for source and pip-wheel simulator installations."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest

from forcebench.registry import task_names
from scripts.expert import parallel_sort_collection


ROOT = Path(__file__).resolve().parents[1]
GENERATORS = {
    "weight_sort": "generate_parallel_sort.py", "rattle_sort": "generate_parallel_sort.py",
    "stove_turn_on": "generate_parallel_stove.py",
    "radmanso_drawer_sort_table": "generate_parallel_radmanso.py",
    "nut_thread": "generate_parallel_privileged_placement.py",
    "faucet_turn": "generate_parallel_faucet.py",
    "faucet_force_turn": "generate_parallel_faucet_force_turn.py",
    "stove_press_turn": "generate_parallel_stove_press_turn.py",
}


def prepare_parallel(tmp_path, monkeypatch, task):
    code = tmp_path / "wheel checkout"
    generator = code / "scripts/expert" / GENERATORS[task]
    generator.parent.mkdir(parents=True)
    generator.write_text("# CPU routing fixture\n")
    monkeypatch.setattr(parallel_sort_collection, "CODE_ROOT", code)
    monkeypatch.setattr(parallel_sort_collection, "SCRATCH_ROOT", tmp_path / "scratch")
    monkeypatch.delenv("FORCEBENCH_ISAACLAB_ROOT", raising=False)
    monkeypatch.delenv("FORCEBENCH_SIM_PYTHON", raising=False)
    calls = []
    def process(command, **kwargs):
        calls.append((command, kwargs))
        return 1, False, 0.01
    monkeypatch.setattr(parallel_sort_collection, "_run_process", process)
    args = parallel_sort_collection.parser().parse_args([
        "--task", task, "--seeds", "7,12", "--max-attempts", "1", "--timeout-s", "123",
        "--dataset-dir", str(tmp_path / "data/datasets" / task), "--run-name", "wheel_test",
    ])
    return code, generator, args, calls


@pytest.mark.parametrize("task", task_names())
def test_wheel_parallel_uses_current_python_and_preserves_collection_contract(tmp_path, monkeypatch, task):
    code, generator, args, calls = prepare_parallel(tmp_path, monkeypatch, task)
    monkeypatch.setenv("PYTHONPATH", "prior-import-path")
    result = parallel_sort_collection.collect(args)
    assert result["failed_seeds"] == [7, 12]
    command, options = calls[0]
    assert command[:2] == [sys.executable, str(generator)]
    assert "-p" not in command
    assert options["cwd"] == code
    assert options["timeout_s"] == 123.0
    assert options["environment"]["PYTHONPATH"] == str(code) + os.pathsep + "prior-import-path"
    assert options["environment"]["FORCEBENCH_TASK"] == task
    seeds = command[command.index("--seeds") + 1].split(",")
    assert len(seeds) == len(set(seeds)) == 32
    assert seeds[:2] == ["7", "12"]
    assert command[command.index("--write_seeds") + 1] == "7,12"
    assert command[command.index("--device") + 1] == "cuda:0"
    assert "--headless" in command and "--enable_cameras" in command
    assert command[command.index("--main_image_size") + 1] == "224"
    if task == "nut_thread":
        assert command[command.index("--trajectory_mode") + 1] == "all_expert"


def test_parallel_preserves_automatic_existing_source_checkout(tmp_path, monkeypatch):
    code, generator, args, calls = prepare_parallel(tmp_path, monkeypatch, "weight_sort")
    source = code / "runtime/IsaacLab"
    source.mkdir(parents=True)
    launcher = source / "isaaclab.sh"
    launcher.write_text("#!/bin/sh\n")
    parallel_sort_collection.collect(args)
    assert calls[0][0][:3] == [str(launcher), "-p", str(generator)]
    assert calls[0][1]["cwd"] == source.resolve()


def test_explicit_python_selects_wheel_even_with_old_source_configuration(tmp_path, monkeypatch):
    code, generator, args, calls = prepare_parallel(tmp_path, monkeypatch, "weight_sort")
    python = str(tmp_path / "sim environment/bin/python")
    monkeypatch.setenv("FORCEBENCH_ISAACLAB_ROOT", str(tmp_path / "retired-source"))
    monkeypatch.setenv("FORCEBENCH_SIM_PYTHON", python)
    parallel_sort_collection.collect(args)
    assert calls[0][0][:2] == [python, str(generator)]
    assert calls[0][1]["cwd"] == code


def test_explicit_missing_source_fails_instead_of_silent_runtime_change(tmp_path, monkeypatch):
    _, _, args, calls = prepare_parallel(tmp_path, monkeypatch, "weight_sort")
    monkeypatch.setenv("FORCEBENCH_ISAACLAB_ROOT", str(tmp_path / "missing-source"))
    with pytest.raises(FileNotFoundError, match="Isaac Lab launcher is missing"):
        parallel_sort_collection.collect(args)
    assert not calls


def find_bash():
    if os.name == "nt":
        git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        return str(git_bash) if git_bash.is_file() else None
    return shutil.which("bash")


def bash_path(bash, path):
    if os.name != "nt":
        return str(path)
    return subprocess.check_output([bash, "-c", 'cygpath -u -- "$1"', "path", str(path)],
                                   text=True, encoding="utf-8").strip()


@pytest.mark.parametrize("runtime_mode", ["wheel", "explicit-wheel", "source"])
@pytest.mark.parametrize("mode,arguments,script", [
    ("expert", ["--task", "weight_sort", "--seeds", "7,12"], "scripts/expert/collect_expert.py"),
    ("policy-vector", ["300", "2", "auto", "out with spaces"], "scripts/evaluation/inference_forcebench_vectorized.py"),
    ("replay", ["episode with spaces.hdf5", "preview with spaces"], "scripts/evaluation/replay_forcebench.py"),
    ("policy-replay", ["rollout with spaces", "movie with spaces.mp4"], "scripts/evaluation/render_policy_rollout.py"),
])
def test_actual_shell_routes_four_modes_without_a_source_dependency(
        tmp_path, runtime_mode, mode, arguments, script):
    source_mode = runtime_mode == "source"
    bash = find_bash()
    if bash is None:
        pytest.skip("bash is required for the actual shell routing check")
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    shutil.copyfile(ROOT / "run_native.sh", checkout / "run_native.sh")
    # Assert all dispatched files exist in the actual release, then keep the
    # test's temporary layout independent of any local simulator installation.
    for name in (script, "scripts/evaluation/resolve_task_max_steps.py"):
        assert (ROOT / name).is_file()
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# routing target\n")
    bin_dir = tmp_path / "fake interpreter with spaces"
    bin_dir.mkdir()
    stub = '''#!/usr/bin/env bash
set -eu
printf '%s\\0' "$PWD" "${PYTHONPATH:-}" "${FORCEBENCH_ASSET_DIR:-}" "$@" >> "$CALL_FILE"
printf '\\n' >> "$CALL_FILE"
if [[ "${1:-}" == */resolve_task_max_steps.py ]]; then printf '17\\n'; fi
'''
    python = bin_dir / "python3"
    python.write_text(stub, encoding="utf-8", newline="\n")
    python.chmod(0o755)
    record = tmp_path / "calls.bin"
    conda_record = tmp_path / "conda_calls.txt"
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("FORCEBENCH_"):
            env.pop(name)
    env.update(CALL_FILE=bash_path(bash, record),
               FORCEBENCH_RUNTIME_ROOT=bash_path(bash, tmp_path / "no bundled runtime"),
               FORCEBENCH_ASSET_DIR=bash_path(bash, tmp_path / "assets with spaces"),
               PYTHONPATH="prior-import-path")
    if source_mode:
        source = tmp_path / "old source with spaces"
        source.mkdir()
        launcher = source / "isaaclab.sh"
        launcher.write_text(stub, encoding="utf-8", newline="\n")
        launcher.chmod(0o755)
        conda = tmp_path / "conda setup.sh"
        conda.write_text('conda() { printf "%s\\n" "$*" >> "$CONDA_CALL_FILE"; }\n',
                         encoding="utf-8", newline="\n")
        env.update(FORCEBENCH_ISAACLAB_ROOT=bash_path(bash, source),
                   FORCEBENCH_CONDA_SH=bash_path(bash, conda),
                   FORCEBENCH_CONDA_ENV="existing-sim", CONDA_CALL_FILE=bash_path(bash, conda_record))
    else:
        # The wheel route must not require or source an absent Conda runtime.
        env["FORCEBENCH_CONDA_SH"] = bash_path(bash, tmp_path / "missing conda.sh")
        if runtime_mode == "explicit-wheel":
            env["FORCEBENCH_SIM_PYTHON"] = bash_path(bash, python)
            env["FORCEBENCH_ISAACLAB_ROOT"] = bash_path(bash, tmp_path / "retired source")
    result = subprocess.run([bash, "-c", 'export PATH="$1:$PATH"; shift; exec bash "$@"',
                             "launcher-test", bash_path(bash, bin_dir),
                             bash_path(bash, checkout / "run_native.sh"), mode, *arguments],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    calls = [line.rstrip(b"\0").decode("utf-8").split("\0") for line in record.read_bytes().splitlines()]
    prefix = ["-p"] if source_mode and mode != "expert" else []
    final = calls[-1]
    expected_cwd = bash_path(bash, source if source_mode else checkout)
    assert final[0] == expected_cwd
    assert final[1].startswith(bash_path(bash, checkout) + ":")
    assert final[1].endswith("prior-import-path")
    assert final[2] == env["FORCEBENCH_ASSET_DIR"]
    assert final[3:3+len(prefix)] == prefix
    assert final[3+len(prefix)] == bash_path(bash, checkout / script)
    forwarded = final[4+len(prefix):]
    if mode == "expert":
        assert forwarded == arguments
    elif mode == "policy-vector":
        assert len(calls) == 2
        assert calls[0][3] == bash_path(bash, checkout / "scripts/evaluation/resolve_task_max_steps.py")
        assert forwarded[forwarded.index("--max_steps") + 1] == "17"
        assert forwarded[forwarded.index("--num_envs") + 1] == "2"
        assert forwarded[forwarded.index("--output_dir") + 1] == "out with spaces"
    elif mode == "replay":
        assert forwarded[0] == arguments[0]
        assert forwarded[forwarded.index("--preview_dir") + 1] == arguments[1]
    else:
        assert forwarded[forwarded.index("--rollout_dir") + 1] == arguments[0]
        assert forwarded[forwarded.index("--video_path") + 1] == arguments[1]
    if source_mode:
        assert conda_record.read_text().strip() == "activate existing-sim"
    else:
        assert not conda_record.exists()
