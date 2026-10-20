"""Coordinate one training output across Lightning/torchrun local DDP ranks.

This module is deliberately independent of torch so its real initialization can
be exercised with CPU subprocesses before starting an expensive training job.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid


TOKEN_ENV = "CORIBENCH_FORCEFLOW_RUN_TOKEN"
MARKER_NAME = ".forceflow-run.json"
_RUN_LOCKS = []  # Keep the primary rank's advisory lock until process exit.


@dataclass(frozen=True)
class RunOutput:
    output: Path
    rank: int
    token: str
    config_sha256: str

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def _process_identity(pid: int) -> str:
    """PID plus process creation time prevents accidental PID-reuse matches."""
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        # The comm field can contain spaces/parentheses; fields after ')' start
        # at field 3, making field 22 (starttime) index 19 of this suffix.
        fields = stat.read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            raise ProcessLookupError(pid)
        start = fields[19]
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    else:
        start = subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "lstart="], text=True
        ).strip()
        if not start:
            raise ProcessLookupError(pid)
        boot = "non-linux"
    return f"{socket.gethostname()}:{boot}:{pid}:{start}"


def _rank_and_token() -> tuple[int, str]:
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if "TORCHELASTIC_RUN_ID" in os.environ:
        world = int(os.environ.get("WORLD_SIZE", "1"))
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
        if local_world != world:
            raise RuntimeError("This training entry supports one-node torchrun; multi-node output initialization is unsupported.")
        # Every worker of this single-node torchrun has the same live launcher.
        # A later torchrun may reuse run_id='none' and MASTER_PORT, but it cannot
        # have the same PID+creation time. Restarts are separate attempts.
        identity = {
            "launcher": _process_identity(os.getppid()),
            "restart": os.environ.get("TORCHELASTIC_RESTART_COUNT", "0"),
            "elastic_id": os.environ["TORCHELASTIC_RUN_ID"],
            "world_size": world,
        }
        token = "torchrun-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    elif rank == 0:
        # Do not reuse an inherited token for an independent primary invocation.
        # Lightning's reexecuted nonzero children inherit this fresh token.
        token = "lightning-" + uuid.uuid4().hex
        os.environ[TOKEN_ENV] = token
    else:
        token = os.environ.get(TOKEN_ENV, "")
        if not token:
            raise RuntimeError("Nonzero rank has no run token from its Lightning parent or torchrun launcher.")
    return rank, token


def _owner_is_alive(marker: dict) -> bool:
    try:
        return _process_identity(int(marker["owner_pid"])) == marker["owner_identity"]
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return False


def initialize_run_output(output: Path, config: dict, *, resume=None,
                          timeout: float = 30.0) -> RunOutput:
    """Create once on rank 0; other ranks may join only that same live run."""
    output = Path(output).resolve()
    rank, token = _rank_and_token()
    fingerprint = hashlib.sha256(json.dumps(
        {"config": config, "output": str(output),
         "resume": None if resume is None else str(Path(resume).resolve())},
        sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()
    result = RunOutput(output, rank, token, fingerprint)
    marker_path = output / MARKER_NAME
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=resume is not None)
        except FileExistsError as error:
            raise FileExistsError(f"Training output already exists: {output}; choose a new output_dir.") from error
        lock = (output / ".forceflow-run.lock").open("a+")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lock.close()
            raise RuntimeError(f"Training output is already owned by a live run: {output}") from error
        _RUN_LOCKS.append(lock)
        (output / "checkpoints").mkdir(exist_ok=True)
        marker = {"version": 1, "token": token, "config_sha256": fingerprint,
                  "owner_pid": os.getpid(), "owner_identity": _process_identity(os.getpid()),
                  "resume": None if resume is None else str(Path(resume).resolve())}
        temporary = output / f".forceflow-run-{uuid.uuid4().hex}.tmp"
        with temporary.open("x") as stream:
            json.dump(marker, stream, sort_keys=True)
            stream.write("\n")
        temporary.replace(marker_path)
        return result

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker_path.is_file():
            marker = json.loads(marker_path.read_text())
            if marker.get("token") != token:
                if resume is not None and not _owner_is_alive(marker):
                    # A resumed worker may start before rank 0 replaces the old
                    # finished attempt's marker. It must never join that marker.
                    time.sleep(0.05)
                    continue
                raise FileExistsError(f"Output belongs to a different training run: {output}")
            if marker.get("config_sha256") != fingerprint:
                raise ValueError(f"DDP ranks disagree on resolved training configuration: {output}")
            if not _owner_is_alive(marker):
                raise RuntimeError(f"Output primary rank is no longer alive: {output}")
            return result
        time.sleep(0.05)
    raise TimeoutError(f"Rank {rank} timed out waiting for its primary rank to initialize {output}")
