# ForceVLA source candidates

**Status: archived source candidates with explicit portable entry points and a bounded real-data GPU diagnostic. Final paper recipe/checkpoint selection and clean installation are not established.** No dataset, normalization values, base model or trained checkpoint is included.

This package preserves the common ForceVLA numerical core: the pi0 backbone, one current six-axis wrench concatenated with seven proprioceptive values, force-aware mixture of experts, two RGB streams, 50 predicted/executed actions and 15 Hz control. It is separate from the controlled pi0.5 F0–F4 family.

## Recorded source and recipe choices

Select a source snapshot explicitly:

| Source snapshot | Preserved difference |
| --- | --- |
| `base` | Base recorded loader and server |
| `stove_sep14` | LeRobot `video_backend="pyav"`; additional 8D transform branch exists but the candidate recipes use 7D actions |
| `stovepress_sep13` | Original websocket timeout 600 seconds; contains the newer Stove Press recipe |

The numerical model, force transforms, optimizer and weight loader are byte-identical across the snapshots. Their training scripts differ only in checkpoint lock placement before packaging. The archived source labels alone do not prove which snapshot produced a paper result.

[recipe_catalog.json](./recipe_catalog.json) lists nine recorded recipe candidates, covering eight task IDs:

| Recipe | Task | Global batch | FSDP devices | Save interval |
| --- | --- | --- | --- | --- |
| `weight_aug24` | Weight Sort | 64 | 4 | 5,000 |
| `rattle_aug24` | CoM Sort (`rattle_sort`) | 64 | 2 | 5,000 |
| `drawer_aug25` | Drawer Pull | 64 | 2 | 5,000 |
| `faucet_aug25` | Faucet Turn (`faucet_turn`) | 64 | 2 | 5,000 |
| `stove_aug25` | Stove Turn | 64 | 2 | 5,000 |
| `nut_aug25` | Nut Thread | 64 | 2 | 5,000 |
| `stovepress_aug29` | Stove Press | 64 | 4 | 5,000 |
| `faucetforce_aug30` | Faucet Direction (`faucet_force_turn`) | 64 | 4 | 5,000 |
| `stovepress_sep13` | Stove Press | 64 | 2 | 10,000 |

These are **candidate recipes, not a confirmed paper-to-run mapping**. No recipe is chosen automatically. The newer Stove Press recipe requires its corresponding source snapshot. Recipe constructors were extracted from the recorded configurations, replacing only resource paths and adding candidate metadata. The newer recipe retains its recorded dataclass overrides of the earlier configuration.

All listed recipes train for 10,000 updates with seed 42 and full parameter tuning. Their inherited learning-rate schedule has warmup 1,000 and decay horizon **30,000**, not 10,000; peak/end rates are 2.5e-5/2.5e-6. AdamW uses the recorded defaults and EMA is 0.99. FSDP size is preserved per recipe and is not silently reduced for a smaller machine.

Normalization is **z-score**, over the 32-dimensional padded concatenation of state+wrench and 32-dimensional padded actions. XYZ/RPY action deltas are relative to the current state; angular deltas are wrapped and gripper targets remain absolute. No temporal force window or future-force supervision is introduced.

## Explicit local inputs

```bash
python baselines/forcevla/run.py check \
  --source-snapshot stovepress_sep13 --recipe stovepress_sep13 \
  --lerobot-home /path/to/lerobot --repo-id namespace/selected_dataset \
  --norm-file /path/to/selected/norm_stats.json \
  --base-params /path/to/pi0_base/params --run-dir /path/to/new_run
```

`check` loads metadata only, creates no run outputs and does not load model weights. It checks 200 episodes, 15 Hz, seven-dimensional state/actions, six-dimensional wrench, two image streams and state/action norm structure. Metadata compatibility is not verification of the actual dataset payload or base weights.

After selecting a verified dataset/norm/base and installing a compatible Linux GPU environment, `run.py train` accepts the same arguments. Set `CUDA_VISIBLE_DEVICES` explicitly to the devices needed by the selected recipe. Cache, output and checkpoint-lock paths are placed under `--run-dir`. Existing checkpoint roots are rejected. The original training loop, save cadence and numerical updates are preserved; checkpoint identities are written after completed saves. Resume is intentionally not exposed until its data-order behavior has been verified.

## Serving

```bash
CUDA_VISIBLE_DEVICES=0 python baselines/forcevla/serve.py \
  --source-snapshot stovepress_sep13 --recipe stovepress_sep13 \
  --lerobot-home /path/to/lerobot --repo-id namespace/selected_dataset \
  --norm-file /path/to/selected/norm_stats.json \
  --base-params /path/to/pi0_base/params --run-dir /path/to/inference_run \
  --checkpoint /path/to/explicit/completed_checkpoint \
  --server-mode single --port 8000
```

Use the repository's `forcevla_ws` adapter for the flat image/state/wrench request. The candidate also exposes the recorded batched server. Batched serving requires explicit `--server-mode batched --host 0.0.0.0 --max-batch-size N --batch-window-ms M`; batch settings affect inference and must come from the matching evaluation record. These production serving modes have not been GPU-validated during this packaging work. A separate diagnostic passed real-model inference and the OpenPI loopback WebSocket handler; that does not validate the candidate's production server or batching behavior.

Serving requires a checkpoint identity matching source snapshot, complete packaged source hash, recipe, input metadata, normalization and completed training budget. The original norm bytes are retained after checking the values saved by OpenPI. There is no identity bypass or automatic checkpoint selection. Historical checkpoints without this candidate's receipt require their original frozen runtime; do not manufacture a receipt to make one load. Base parameter identity currently includes its resolved path, so relocating it requires an explicit provenance decision.

## Environment and licenses

Each `runtime_*` preserves source, OpenPI client, `pyproject.toml`, `uv.lock`, the Apache license and Flaxformer source/license. Missing Flaxformer license/setup companions were recovered read-only from the original installation. [SOURCE_MANIFEST.json](./SOURCE_MANIFEST.json) records all 680 source files and [PACKAGED_SOURCE_MANIFEST.json](./PACKAGED_SOURCE_MANIFEST.json) records copied and modified files. Git attributes retain exact source bytes.

The original dependency files are **references, not a clean-install recipe**: LeRobot was installed separately, and the observed current environment differs from the pyproject in packages such as Transformers. [ENVIRONMENT_REFERENCE.json](./ENVIRONMENT_REFERENCE.json) records observed versions without claiming they are the historical training environment. The Flaxformer source is on the wrapper's import path; unrelated RLDS dependencies are not exercised by these LeRobot recipes. The concrete dependency reconciliation is to recover the selected run's LeRobot source/version (the original pyproject does not declare it) and resolve Transformers 4.48.1 in the pyproject versus 4.53.2 in the observed environment. Client dependencies are declared separately under `packages/openpi-client`. A clean installation for this variant remains necessary.

On 2026-10-21, the explicit `base` / `faucet_aug25` candidate passed a bounded diagnostic on two H200 GPUs in an existing compatible environment (Python 3.11, JAX 0.5.3, Transformers 4.53.2, LeRobot 0.1.0). It decoded real observations from a 200-episode Faucet dataset, loaded the actual pi0 base parameters, ran three full numerical updates at the preserved global batch 64 / FSDP 2, and produced finite `[50,7]` actions directly and through the OpenPI loopback WebSocket handler. Other source/recipe combinations, full training, production checkpoint restoration and paper scores were not tested. The optional repository tool `tools/gpu_smoke_openpi.py --family forcevla --method forcevla --source-snapshot base --recipe faucet_aug25` requires explicit input paths; see `--help`. It saves no production checkpoint or identity receipt.

## Verification and remaining work

```bash
python baselines/forcevla/validate_candidate.py
python baselines/forcevla/audit_candidate.py --source-root /path/to/source_snapshots --check-transforms
```

The audit validates original hashes and preserved cores, then exercises actual NumPy input/output, delta/absolute-action, angle-wrap and z-score definitions without loading a model. The wrapper check exercises metadata CLI and checkpoint identity/asset handling with fixtures; it does not start a server or train.

Before declaring a paper-aligned release, resolve: the per-task final source/recipe/checkpoint mapping; full dataset and base-weight checksums; matching norm provenance; exact inference batching; clean dependency installation; and full GPU/simulator evaluation. CPU success does not prove paper-score reproduction.
