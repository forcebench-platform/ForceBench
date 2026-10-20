# Controlled pi0.5 F0–F4 candidate

**Status: portable source candidate; CPU checks passed, clean installation and GPU execution not yet validated.** This package does not establish the final runtime/checkpoint selection for all forty paper task–method cells.

## Methods and scope

| Historical ID | Paper system |
| --- | --- |
| F0 | pi0.5 (No Force) |
| F1 | Token Fusion |
| F2 | Prompt Fusion |
| F3 | Cross-Attention |
| F4 | Token Fusion + Pred |

The public entry points accept only the eight paper environment IDs: `weight_sort`, `rattle_sort`, `faucet_turn`, `faucet_force_turn`, `stove_turn_on`, `stove_press_turn`, `radmanso_drawer_sort_table`, and `nut_thread`.

The numeric model, input/output transforms, data loader, optimizer and training update implementation come from the September 10 controlled runtime. The preserved recipe uses pi0.5, 10,000 updates, batch 32, seed 42, 50 predicted/executed actions and 15 Hz control. F4 predicts ten future wrench samples during training with auxiliary weight 0.0003. These are candidate implementation settings, not proof of each historical paper row.

## Source evidence

- [SOURCE_MANIFEST.json](./SOURCE_MANIFEST.json): 137 source files across the September 10 and September 14 Nut snapshots, with relative paths, sizes and hashes.
- Both complete historical source identities match their recorded values: `57961537...65afedb` and `c0bd8593...6a25d`, respectively; each hashes 68 files. The model, model configuration, ForceBench transforms and optimizer are byte-identical across those snapshots.
- Nine preserved upstream files match OpenPI commit `215abfb217dbac7d5f1273282331b9b1866c0479`.
- The missing license, client and dependency reference files were recovered read-only from the original runtime's upstream reference. The active interpreter's OpenPI client matches that reference byte-for-byte.
- [PACKAGED_SOURCE_MANIFEST.json](./PACKAGED_SOURCE_MANIFEST.json) records source hashes before portability edits and packaged hashes. [ENVIRONMENT_REFERENCE.json](./ENVIRONMENT_REFERENCE.json) records the observed Python 3.11.11 environment; it is not a fresh-install receipt.
- Local Git attributes preserve exact file bytes, including captured mixed line endings, so checkout conversion does not silently invalidate those hashes.
- Apache and Gemma license texts are preserved under [runtime](./runtime/). Upstream dependency metadata and lock are preserved; their compatibility with this packaged candidate still needs a clean Linux/GPU installation check.

No dataset, normalization values, base weights or trained checkpoint is bundled. InternVLA and the old Gear/Peg training registry are outside the entry-point task choices.

## Explicit inputs

Both entry points require task, method, LeRobot root, dataset ID, normalization file, base parameter directory and separate run directory. Serving additionally requires an explicit checkpoint. No checkpoint is chosen by timestamp or directory order.

`run.py check` is a CPU/standard-library metadata check and does not load weights or create a run directory:

```bash
python baselines/pi05/run.py check \
  --task weight_sort --method F0 \
  --lerobot-home /path/to/lerobot --repo-id namespace/selected_dataset \
  --norm-file /path/to/selected/norm_stats.json \
  --base-params /path/to/pi05_base/params --run-dir /path/to/new_run
```

It checks the 200-episode/15 Hz metadata, feature shapes and method-specific normalization structure. It records metadata and norm hashes, but does **not** validate dataset payload checksums or base parameter contents. Select inputs from a verified per-task provenance manifest; a metadata-compatible dataset is not automatically the paper dataset.

The candidate also implements `run.py train` with the same arguments and `serve.py --checkpoint ...`. Their CLI parsing and identity gates are checked locally; **the training, model-loading and serving paths have not been run during packaging**. GPU execution requires an explicit `CUDA_VISIBLE_DEVICES`. Training optionally supports the preserved three-update `--preflight` and identity-checked `--resume`; no job is started by importing the CLI modules.

Training copies the supplied norm file byte-for-byte into the new run's assets. After OpenPI saves a checkpoint, the wrapper checks that its normalization values match and preserves the exact imported bytes before writing the checkpoint identity; JSON reformatting therefore cannot invalidate an otherwise matching checkpoint. It does not recompute or adjust quantiles. Historical normalization builders and the original assets-copy provenance were recovered for further audit, but do not yet define a verified norm-generation recipe for every final task/model pair.

## Identity and archived checkpoints

The portable schema is `forcebench_portable_pi05_candidate_20261014`. Serving requires matching schema, task, method, complete packaged source identity, explicit input identity, normalization hash and 10,000 completed updates. Missing or mismatched identity is an error; there is no diagnostic override.

Portability edits change source identity. **Historical checkpoints therefore require their corresponding frozen runtime** and will not pass this new candidate's guard. Do not edit the checkpoint receipt or disable verification to make an archived checkpoint load.

The new source guard includes runtime source/scripts/client, dependency metadata and portable Python wrappers. Environment caches, installed virtual environments and output receipts are excluded. The input identity currently includes the resolved base-parameter path; relocating it requires an explicit provenance/compatibility decision rather than silently treating a different directory as the same base.

## Verification

```bash
python baselines/pi05/validate_candidate.py
python baselines/pi05/audit_candidate.py --source-root /path/to/source_snapshots --check-transforms
```

The first command checks the packaged source; the second checks the separately supplied original snapshots. Both use NumPy for isolated transform checks. They do not start training or a websocket server.

[PORTABLE_VALIDATION.json](./PORTABLE_VALIDATION.json) records:

- Python syntax checks and dependency-light CLI checks;
- real, unmodified NumPy transform definitions: F0–F4 force routing, two-camera masks, training-only future labels, F4 padding, output horizon, angle wrapping and wrench clipping;
- rejection of missing/mismatched checkpoint identities, wrong source or norm hashes, incomplete training budget and dataset path traversal.

The transform definitions are AST-isolated to avoid model/dependency imports. No replacement tokenizer or model is used; HWC image behavior is checked, while CHW/einops is not exercised. These checks do not establish normalized train/serve parity, gradients, checkpoint loading, full inference, fresh installation or paper score reproduction. Those remain release gates.
