"""Standard, restart-safe supervision for task-bound expert backends."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .expert_api import EXPERT_COLLECTION_CONTRACT, ExpertRequest
from .expert_registry import get_expert_backend_manifest, load_expert_backend
from .paths import CODE_ROOT, DATA_ROOT, SCRATCH_ROOT
from .recording import parse_success_episode_stem, validate_hdf5_episode
from .registry import get_task_manifest


RAW_WRENCH_FIELD = "forces/control/eef_wrench_joint_raw"
EMA_WRENCH_FIELD = "forces/control/eef_wrench_joint_ema"
RECORDED_WRENCH_FIELDS = (RAW_WRENCH_FIELD, EMA_WRENCH_FIELD)
PERSISTENT_EXPERT_BACKENDS = frozenset()
_SEED_RANGE = re.compile(r"^(\d+)-(\d+)$")
_SEED_SLICE = re.compile(r"^(\d+):(\d+)(?::(\d+))?$")
_PROCESS_TERMINATE_GRACE_S = 10.0
_PROCESS_KILL_GRACE_S = 10.0
_PROCESS_GROUP_POLL_S = 0.05


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_seed_spec(value: str) -> tuple[int, ...]:
    """Parse ``1,4-6,10:16:2`` into sorted unique non-negative seeds."""

    seeds: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"invalid empty seed expression in {value!r}")
        if part.isdigit():
            seeds.add(int(part))
            continue
        range_match = _SEED_RANGE.fullmatch(part)
        if range_match is not None:
            start, end = (int(item) for item in range_match.groups())
            if end < start:
                raise ValueError(f"seed range must be ascending: {part!r}")
            seeds.update(range(start, end + 1))
            continue
        slice_match = _SEED_SLICE.fullmatch(part)
        if slice_match is not None:
            start = int(slice_match.group(1))
            stop = int(slice_match.group(2))
            step = int(slice_match.group(3) or 1)
            if stop <= start or step < 1:
                raise ValueError(f"seed slice must have stop > start and step > 0: {part!r}")
            seeds.update(range(start, stop, step))
            continue
        raise ValueError(f"invalid seed expression: {part!r}")
    if not seeds:
        raise ValueError("at least one seed is required")
    return tuple(sorted(seeds))


def _inside(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def validate_storage_boundaries(dataset_dir: Path, run_dir: Path) -> None:
    dataset_dir = dataset_dir.resolve()
    run_dir = run_dir.resolve()
    formal_root = (DATA_ROOT / "datasets").resolve()
    expert_scratch_root = (SCRATCH_ROOT / "expert").resolve()
    if not _inside(dataset_dir, formal_root) or dataset_dir == formal_root:
        raise ValueError(f"expert HDF5 output must be below {formal_root}: {dataset_dir}")
    if not _inside(run_dir, expert_scratch_root) or run_dir == expert_scratch_root:
        raise ValueError(f"expert logs/videos must be below {expert_scratch_root}: {run_dir}")
    if dataset_dir == run_dir or _inside(dataset_dir, run_dir) or _inside(run_dir, dataset_dir):
        raise ValueError("formal expert data and scratch artifacts must be separate")


def successful_episodes(dataset_dir: Path, task: str) -> dict[int, tuple[Path, ...]]:
    """Return canonical successful episodes, rejecting a mixed/invalid dataset."""

    grouped: dict[int, list[Path]] = {}
    if not dataset_dir.exists():
        return {}
    for path in sorted(dataset_dir.glob("seed_*_episode_*.hdf5")):
        parsed = parse_success_episode_stem(path.stem)
        if parsed is None:
            raise ValueError(f"non-canonical expert episode name: {path}")
        report = validate_hdf5_episode(path)
        if report["task"] != task:
            raise ValueError(
                f"dataset {dataset_dir} mixes tasks: expected {task!r}, "
                f"found {report['task']!r} in {path.name}"
            )
        seed = int(report["seed"])
        if seed != parsed[0]:
            raise ValueError(f"episode seed metadata/name mismatch: {path}")
        grouped.setdefault(seed, []).append(path.resolve())
    return {seed: tuple(paths) for seed, paths in grouped.items()}


@dataclass(frozen=True)
class ExpertCollectionPlan:
    task: str
    backend: str
    seeds: tuple[int, ...]
    dataset_dir: Path
    run_dir: Path
    isaaclab_root: Path
    devices: tuple[str, ...] = ("cuda:0",)
    workers: int = 1
    episodes_per_seed: int = 1
    max_attempts: int = 2
    timeout_s: float = 900.0
    launch_stagger_s: float = 20.0
    main_image_size: int = 224
    wrist_image_size: int = 224
    record_videos: bool = False
    persistent_workers: bool = True
    seeds_per_worker_process: int = 8
    checkpoint: Path | None = None
    backend_options: Mapping[str, Any] = field(default_factory=dict)
    code_root: Path = CODE_ROOT

    def validate(self) -> None:
        get_task_manifest(self.task)
        manifest = get_expert_backend_manifest(self.task, self.backend)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("expert seeds must be unique and non-negative")
        if (
            self.workers < 1
            or self.episodes_per_seed < 1
            or self.max_attempts < 1
            or self.seeds_per_worker_process < 1
        ):
            raise ValueError(
                "workers, episodes-per-seed, max-attempts and "
                "seeds-per-worker-process must be positive"
            )
        if self.timeout_s <= 0.0 or self.launch_stagger_s < 0.0:
            raise ValueError("timeout must be positive and launch stagger cannot be negative")
        if not self.devices or any(not device for device in self.devices):
            raise ValueError("at least one non-empty simulator device is required")
        if not self.code_root.is_absolute() or not self.isaaclab_root.is_absolute():
            raise ValueError("code and Isaac Lab roots must be absolute")
        if not isinstance(self.record_videos, bool):
            raise TypeError("record-videos must be a boolean")
        if not isinstance(self.persistent_workers, bool):
            raise TypeError("persistent-workers must be a boolean")
        if not (self.isaaclab_root / "isaaclab.sh").is_file():
            raise FileNotFoundError(f"Isaac Lab launcher is missing below {self.isaaclab_root}")
        validate_storage_boundaries(self.dataset_dir, self.run_dir)
        if manifest.requires_checkpoint and self.checkpoint is None:
            raise ValueError(f"expert backend {self.backend!r} requires --checkpoint")


@dataclass(frozen=True)
class AttemptResult:
    seed: int
    target_episode: int
    attempt: int
    command: tuple[str, ...]
    log: str
    return_code: int | None
    timed_out: bool
    new_episodes: tuple[str, ...]
    success: bool
    elapsed_s: float
    batch_size: int = 1
    batch_elapsed_s: float | None = None


@dataclass(frozen=True)
class SeedResult:
    seed: int
    requested: int
    existing_before: int
    valid_after: int
    success: bool
    attempts: tuple[AttemptResult, ...]


def _process_group_exists(process_group_id: int) -> bool:
    """Return whether a dedicated child process group still has members."""

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process: subprocess.Popen,
    process_group_id: int,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        # Reap the group leader as soon as it exits. Otherwise its zombie can
        # make killpg(..., 0) report a non-empty group indefinitely.
        process.poll()
        if not _process_group_exists(process_group_id):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_PROCESS_GROUP_POLL_S)


def _signal_process_group(process_group_id: int, signum: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, signum)
    except ProcessLookupError:
        pass


def _clean_process_group(
    process: subprocess.Popen,
    *,
    process_group_id: int,
    output,
    reason: str,
) -> int | None:
    """Terminate every process in a launch group, including orphaned Isaac children."""

    output.write(
        (f"\n=== {reason}; terminating process group {process_group_id} ===\n").encode()
    )
    output.flush()
    _signal_process_group(process_group_id, signal.SIGTERM)
    if not _wait_for_process_group_exit(
        process,
        process_group_id,
        _PROCESS_TERMINATE_GRACE_S,
    ):
        output.write(
            (
                f"=== process group {process_group_id} survived SIGTERM; "
                "sending SIGKILL ===\n"
            ).encode()
        )
        output.flush()
        _signal_process_group(process_group_id, signal.SIGKILL)
        if not _wait_for_process_group_exit(
            process,
            process_group_id,
            _PROCESS_KILL_GRACE_S,
        ):
            raise RuntimeError(
                f"process group {process_group_id} survived SIGKILL; "
                "collector will not launch another Isaac process"
            )
    process.poll()
    return process.returncode


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    timeout_s: float,
) -> tuple[int | None, bool, float]:
    started = time.monotonic()
    timed_out = False
    return_code: int | None = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as output:
        output.write((f"\n=== launch {utc_now()} ===\n").encode())
        output.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process_group_id = process.pid
        try:
            return_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = _clean_process_group(
                process,
                process_group_id=process_group_id,
                output=output,
                reason=f"timeout after {timeout_s:.1f}s",
            )
        else:
            # IsaacLab's CLI launcher can exit while its simulator child remains
            # alive in the same session. Never begin a retry until that entire
            # group is gone, even when the launcher returned zero.
            process.poll()
            if _process_group_exists(process_group_id):
                return_code = _clean_process_group(
                    process,
                    process_group_id=process_group_id,
                    output=output,
                    reason=f"launcher exited with code {return_code} but descendants remain",
                )
    return return_code, timed_out, time.monotonic() - started


def _request(
    plan: ExpertCollectionPlan,
    *,
    seed: int,
    attempt: int,
    device: str,
    artifact_dir: Path,
) -> ExpertRequest:
    return ExpertRequest(
        task=plan.task,
        backend=plan.backend,
        seed=seed,
        attempt=attempt,
        code_root=plan.code_root.resolve(),
        isaaclab_root=plan.isaaclab_root.resolve(),
        dataset_dir=plan.dataset_dir.resolve(),
        artifact_dir=artifact_dir.resolve(),
        device=device,
        main_image_size=plan.main_image_size,
        wrist_image_size=plan.wrist_image_size,
        record_video=plan.record_videos,
        checkpoint=plan.checkpoint.resolve() if plan.checkpoint is not None else None,
        options=dict(plan.backend_options),
    )


def _persistent_worker_command(
    plan: ExpertCollectionPlan,
    requests: Sequence[ExpertRequest],
    *,
    jobs_path: Path,
    device: str,
) -> list[str]:
    """Build one Isaac launch that executes several existing seed commands."""

    if not requests:
        raise ValueError("persistent expert worker requires at least one request")
    backend = load_expert_backend(plan.task, plan.backend)
    jobs: list[dict[str, Any]] = []
    for request in requests:
        command = backend.build_command(request)
        expected_prefix = [str(plan.isaaclab_root / "isaaclab.sh"), "-p"]
        if command[:2] != expected_prefix or len(command) < 3:
            raise RuntimeError(
                f"expert backend {plan.backend!r} returned a non-Isaac command"
            )
        jobs.append(
            {"seed": request.seed, "script": command[2], "argv": command[3:]}
        )
    jobs_path.parent.mkdir(parents=True, exist_ok=True)
    jobs_path.write_text(
        json.dumps(
            {"schema": "forcebench_persistent_expert_jobs_v1", "jobs": jobs},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    worker_script = plan.code_root / "scripts/expert/run_persistent_expert_worker.py"
    if not worker_script.is_file():
        raise FileNotFoundError(f"persistent expert worker is missing: {worker_script}")
    return [
        str(plan.isaaclab_root / "isaaclab.sh"),
        "-p",
        str(worker_script),
        "--jobs",
        str(jobs_path),
        "--headless",
        "--enable_cameras",
        "--device",
        device,
        # Isaac Sim 6 can otherwise enable experimental geometry streaming
        # from a persisted user preference.  Its RTX/Hydra delegate warns that
        # this conflicts with Fabric transform reads and can render stale
        # dynamic transforms across long-lived episodes.  This is a startup
        # setting, so it must be supplied to the persistent process here.
        "--kit_args=--/UJITSO/geometry=false --/persistent/UJITSO/geometry=false",
    ]


def _collect_seed_batch(
    plan: ExpertCollectionPlan,
    *,
    seeds: Sequence[int],
    device: str,
    worker_index: int,
    launch_delay_s: float,
) -> tuple[SeedResult, ...]:
    """Collect a shard in bounded persistent Isaac processes per attempt."""

    if launch_delay_s:
        time.sleep(launch_delay_s)
    initial = successful_episodes(plan.dataset_dir, plan.task)
    existing_before = {seed: len(initial.get(seed, ())) for seed in seeds}
    current_count = dict(existing_before)
    attempts_by_seed: dict[int, list[AttemptResult]] = {seed: [] for seed in seeds}

    for attempt in range(1, plan.max_attempts + 1):
        attempt_seeds = [
            seed for seed in seeds if current_count[seed] < plan.episodes_per_seed
        ]
        if not attempt_seeds:
            break
        for batch_index, offset in enumerate(
            range(0, len(attempt_seeds), plan.seeds_per_worker_process), start=1
        ):
            batch_seeds = attempt_seeds[offset : offset + plan.seeds_per_worker_process]
            target_episodes = {seed: current_count[seed] + 1 for seed in batch_seeds}
            requests = []
            for seed in batch_seeds:
                artifact_dir = (
                    plan.run_dir
                    / "artifacts"
                    / f"seed_{seed:03d}"
                    / f"episode_{target_episodes[seed]:02d}"
                    / f"attempt_{attempt:02d}"
                )
                artifact_dir.mkdir(parents=True, exist_ok=True)
                requests.append(
                    _request(
                        plan,
                        seed=seed,
                        attempt=attempt,
                        device=device,
                        artifact_dir=artifact_dir,
                    )
                )
            batch_dir = (
                plan.run_dir
                / "persistent_workers"
                / f"worker_{worker_index:02d}"
                / f"attempt_{attempt:02d}"
                / f"batch_{batch_index:03d}"
            )
            jobs_path = batch_dir / "jobs.json"
            command = _persistent_worker_command(
                plan, requests, jobs_path=jobs_path, device=device
            )
            before = {
                path
                for paths in successful_episodes(plan.dataset_dir, plan.task).values()
                for path in paths
            }
            environment = os.environ.copy()
            environment.update(
                {
                    "ACCEPT_EULA": "Y",
                    "OMNI_KIT_ACCEPT_EULA": "YES",
                    "PYTHONUNBUFFERED": "1",
                    "FORCEBENCH_ASSET_DIR": str((plan.code_root / "assets").resolve()),
                    "FORCEBENCH_TASK": plan.task,
                    "FORCEBENCH_EXPERT_COLLECTION_CONTRACT": EXPERT_COLLECTION_CONTRACT,
                    "FORCEBENCH_EXPERT_BACKEND": plan.backend,
                    "FORCEBENCH_EXPERT_WRENCH_FIELD": RAW_WRENCH_FIELD,
                    "FORCEBENCH_EXPERT_WRENCH_EMA_FIELD": EMA_WRENCH_FIELD,
                }
            )
            log_path = (
                plan.run_dir
                / "logs"
                / (
                    f"persistent_worker_{worker_index:02d}_attempt_{attempt:02d}_"
                    f"batch_{batch_index:03d}.log"
                )
            )
            return_code, timed_out, elapsed_s = _run_process(
                command,
                cwd=plan.isaaclab_root,
                environment=environment,
                log_path=log_path,
                timeout_s=plan.timeout_s * len(batch_seeds),
            )
            after_grouped = successful_episodes(plan.dataset_dir, plan.task)
            after = {path for paths in after_grouped.values() for path in paths}
            for seed in batch_seeds:
                current_count[seed] = len(after_grouped.get(seed, ()))
                new_for_seed = tuple(
                    str(path)
                    for path in sorted(after - before)
                    if parse_success_episode_stem(path.stem)[0] == seed
                )
                attempts_by_seed[seed].append(
                    AttemptResult(
                        seed=seed,
                        target_episode=target_episodes[seed],
                        attempt=attempt,
                        command=tuple(command),
                        log=str(log_path.resolve()),
                        return_code=return_code,
                        timed_out=timed_out,
                        new_episodes=new_for_seed,
                        success=current_count[seed] >= plan.episodes_per_seed,
                        elapsed_s=elapsed_s / len(batch_seeds),
                        batch_size=len(batch_seeds),
                        batch_elapsed_s=elapsed_s,
                    )
                )

    return tuple(
        SeedResult(
            seed=seed,
            requested=plan.episodes_per_seed,
            existing_before=existing_before[seed],
            valid_after=current_count[seed],
            success=current_count[seed] >= plan.episodes_per_seed,
            attempts=tuple(attempts_by_seed[seed]),
        )
        for seed in seeds
    )


def _collect_seed(
    plan: ExpertCollectionPlan,
    *,
    seed: int,
    device: str,
    launch_delay_s: float,
) -> SeedResult:
    if launch_delay_s:
        time.sleep(launch_delay_s)
    backend = load_expert_backend(plan.task, plan.backend)
    initial = successful_episodes(plan.dataset_dir, plan.task)
    existing_before = len(initial.get(seed, ()))
    current_count = existing_before
    attempts: list[AttemptResult] = []
    while current_count < plan.episodes_per_seed:
        target_episode = current_count + 1
        completed = False
        for attempt in range(1, plan.max_attempts + 1):
            artifact_dir = (
                plan.run_dir
                / "artifacts"
                / f"seed_{seed:03d}"
                / f"episode_{target_episode:02d}"
                / f"attempt_{attempt:02d}"
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            request = _request(
                plan,
                seed=seed,
                attempt=attempt,
                device=device,
                artifact_dir=artifact_dir,
            )
            command = backend.build_command(request)
            before = {
                path
                for paths in successful_episodes(plan.dataset_dir, plan.task).values()
                for path in paths
            }
            environment = os.environ.copy()
            environment.update(
                {
                    "ACCEPT_EULA": "Y",
                    "OMNI_KIT_ACCEPT_EULA": "YES",
                    "PYTHONUNBUFFERED": "1",
                    "FORCEBENCH_ASSET_DIR": str((plan.code_root / "assets").resolve()),
                    "FORCEBENCH_TASK": plan.task,
                    "FORCEBENCH_EXPERT_COLLECTION_CONTRACT": EXPERT_COLLECTION_CONTRACT,
                    "FORCEBENCH_EXPERT_BACKEND": plan.backend,
                    "FORCEBENCH_EXPERT_WRENCH_FIELD": RAW_WRENCH_FIELD,
                    "FORCEBENCH_EXPERT_WRENCH_EMA_FIELD": EMA_WRENCH_FIELD,
                }
            )
            log_path = (
                plan.run_dir
                / "logs"
                / f"seed_{seed:03d}_episode_{target_episode:02d}_attempt_{attempt:02d}.log"
            )
            return_code, timed_out, elapsed_s = _run_process(
                command,
                cwd=plan.isaaclab_root,
                environment=environment,
                log_path=log_path,
                timeout_s=plan.timeout_s,
            )
            after_grouped = successful_episodes(plan.dataset_dir, plan.task)
            after = {path for paths in after_grouped.values() for path in paths}
            new_for_seed = tuple(
                str(path) for path in sorted(after - before) if parse_success_episode_stem(path.stem)[0] == seed
            )
            current_count = len(after_grouped.get(seed, ()))
            success = current_count >= target_episode
            attempts.append(
                AttemptResult(
                    seed=seed,
                    target_episode=target_episode,
                    attempt=attempt,
                    command=tuple(command),
                    log=str(log_path.resolve()),
                    return_code=return_code,
                    timed_out=timed_out,
                    new_episodes=new_for_seed,
                    success=success,
                    elapsed_s=elapsed_s,
                )
            )
            if success:
                completed = True
                break
        if not completed:
            break
    return SeedResult(
        seed=seed,
        requested=plan.episodes_per_seed,
        existing_before=existing_before,
        valid_after=current_count,
        success=current_count >= plan.episodes_per_seed,
        attempts=tuple(attempts),
    )


def _append_event(path: Path, event: Mapping[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(dict(event), ensure_ascii=False, separators=(",", ":")) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        print("FORCEBENCH_EXPERT_EVENT=" + line.rstrip(), flush=True)


def execute_expert_collection(
    plan: ExpertCollectionPlan, *, dry_run: bool = False
) -> dict[str, Any]:
    """Execute or preview one standardized expert collection plan."""

    plan.validate()
    backend = load_expert_backend(plan.task, plan.backend)
    existing = successful_episodes(plan.dataset_dir, plan.task)
    pending = [seed for seed in plan.seeds if len(existing.get(seed, ())) < plan.episodes_per_seed]
    use_persistent_workers = bool(
        plan.persistent_workers
        and (plan.task, plan.backend) in PERSISTENT_EXPERT_BACKENDS
        and plan.episodes_per_seed == 1
    )
    if dry_run:
        commands = []
        for index, seed in enumerate(pending):
            request = _request(
                plan,
                seed=seed,
                attempt=1,
                device=plan.devices[index % len(plan.devices)],
                artifact_dir=plan.run_dir / "artifacts" / f"seed_{seed:03d}" / "episode_01" / "attempt_01",
            )
            commands.append({"seed": seed, "command": backend.build_command(request)})
        return {
            "contract": EXPERT_COLLECTION_CONTRACT,
            "dry_run": True,
            "task": plan.task,
            "backend": plan.backend,
            "raw_wrench_field": RAW_WRENCH_FIELD,
            "recorded_wrench_fields": list(RECORDED_WRENCH_FIELDS),
            "record_videos": plan.record_videos,
            "persistent_workers": use_persistent_workers,
            "seeds_per_worker_process": plan.seeds_per_worker_process,
            "dataset_dir": str(plan.dataset_dir.resolve()),
            "run_dir": str(plan.run_dir.resolve()),
            "pending_seeds": pending,
            "commands": commands,
        }

    plan.dataset_dir.mkdir(parents=True, exist_ok=True)
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    (plan.run_dir / "logs").mkdir(parents=True, exist_ok=True)
    manifest_path = plan.run_dir / "manifest.jsonl"
    lock_path = plan.run_dir / "collector.lock"
    event_lock = threading.Lock()
    results: list[SeedResult] = []

    import fcntl

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another expert collector owns {lock_path}") from error
        _append_event(
            manifest_path,
            {
                "event": "collection_started",
                "timestamp": utc_now(),
                "contract": EXPERT_COLLECTION_CONTRACT,
                "task": plan.task,
                "backend": plan.backend,
                "raw_wrench_field": RAW_WRENCH_FIELD,
                "recorded_wrench_fields": list(RECORDED_WRENCH_FIELDS),
                "record_videos": plan.record_videos,
                "persistent_workers": use_persistent_workers,
                "seeds_per_worker_process": plan.seeds_per_worker_process,
                "seeds": list(plan.seeds),
                "pending_seeds": pending,
                "episodes_per_seed": plan.episodes_per_seed,
                "workers": plan.workers,
                "devices": list(plan.devices),
                "dataset_dir": str(plan.dataset_dir.resolve()),
            },
            event_lock,
        )
        with ThreadPoolExecutor(max_workers=plan.workers) as executor:
            if use_persistent_workers:
                worker_count = min(plan.workers, len(pending)) if pending else 0
                shards = [pending[index::worker_count] for index in range(worker_count)]
                futures = {
                    executor.submit(
                        _collect_seed_batch,
                        plan,
                        seeds=shard,
                        device=plan.devices[index % len(plan.devices)],
                        worker_index=index,
                        launch_delay_s=index * plan.launch_stagger_s,
                    ): tuple(shard)
                    for index, shard in enumerate(shards)
                }
            else:
                futures = {
                    executor.submit(
                        _collect_seed,
                        plan,
                        seed=seed,
                        device=plan.devices[index % len(plan.devices)],
                        launch_delay_s=(index % plan.workers) * plan.launch_stagger_s,
                    ): (seed,)
                    for index, seed in enumerate(pending)
                }
            for future in as_completed(futures):
                seeds = futures[future]
                try:
                    completed = future.result()
                    batch_results = completed if isinstance(completed, tuple) else (completed,)
                except Exception as error:
                    batch_results = tuple(
                        SeedResult(
                            seed=seed,
                            requested=plan.episodes_per_seed,
                            existing_before=len(existing.get(seed, ())),
                            valid_after=len(existing.get(seed, ())),
                            success=False,
                            attempts=(),
                        )
                        for seed in seeds
                    )
                    for seed in seeds:
                        _append_event(
                            manifest_path,
                            {
                                "event": "seed_error",
                                "timestamp": utc_now(),
                                "seed": seed,
                                "error": f"{type(error).__name__}: {error}",
                            },
                            event_lock,
                        )
                for result in batch_results:
                    results.append(result)
                    _append_event(
                        manifest_path,
                        {
                            "event": "seed_finished",
                            "timestamp": utc_now(),
                            **asdict(result),
                        },
                        event_lock,
                    )

    final = successful_episodes(plan.dataset_dir, plan.task)
    results.sort(key=lambda item: item.seed)
    summary = {
        "contract": EXPERT_COLLECTION_CONTRACT,
        "task": plan.task,
        "backend": plan.backend,
        "backend_metadata": dict(backend.metadata),
        "raw_wrench_field": RAW_WRENCH_FIELD,
        "recorded_wrench_fields": list(RECORDED_WRENCH_FIELDS),
        "record_videos": plan.record_videos,
        "persistent_workers": use_persistent_workers,
        "seeds_per_worker_process": plan.seeds_per_worker_process,
        "filter": "none",
        "dataset_dir": str(plan.dataset_dir.resolve()),
        "run_dir": str(plan.run_dir.resolve()),
        "requested_seeds": list(plan.seeds),
        "episodes_per_seed": plan.episodes_per_seed,
        "valid_episode_counts": {str(seed): len(paths) for seed, paths in sorted(final.items())},
        "successful_seeds": [item.seed for item in results if item.success],
        "failed_seeds": [item.seed for item in results if not item.success],
        "skipped_complete_seeds": [seed for seed in plan.seeds if seed not in pending],
        "finished_at": utc_now(),
        "results": [asdict(item) for item in results],
    }
    summary_path = plan.run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
