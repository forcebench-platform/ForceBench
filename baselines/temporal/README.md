# Temporal Fusion and Dual-Path Fusion

These portable recipes use the actual model/data implementation from the September 14 Nut Thread/Stove training snapshot. Both use 16 **control-rate** wrist-wrench frames at 15 Hz. Temporal Fusion appends `[current wrench, window mean, population standard deviation]` to the seven-dimensional robot state before pi0.5 state tokenization. Dual-Path Fusion also passes the 16-frame history through the statistics encoder as four continuous force tokens. The model definitions, action transforms, and data-loader history queries are retained.

The entry points preserve the observed 10,000 updates, batch 32, seed 42, 50 predicted/executed actions, quantile normalization, angular wrapping, and near-constant quantile guard. Dual-Path Fusion uses force dropout 0.3. The observed default learning-rate schedule has a **30,000-step decay horizon**, although training stops at 10,000 updates; this was not changed to a 10,000-step decay.

## Scope of provenance

The model source is byte-identical between the September 11 Dual-Path snapshot and the later Nut/Stove snapshot. The latter changed datasets and portable execution configuration. The present factory uses the later single-FSDP-shard, four-worker setup; earlier task launchers sometimes used two shards/eight workers. This packaging does **not** establish that every archived checkpoint used an identical launcher or dataset revision. Use the task/checkpoint provenance manifest when reproducing a reported row.

Only source is included. Base pi0.5 weights, official task datasets, per-run statistics, and trained checkpoints must be supplied separately. No experimental Gear/Peg/Wipe task registry is included.

## Environment

Use a separate Linux/Python 3.11 environment; this modified `openpi` namespace must not be mixed with the pi0 or other baseline runtime:

```bash
cd baselines/temporal/runtime
uv sync --python 3.11
cd ..
export TEMPORAL_PYTHON="$PWD/runtime/.venv/bin/python"
```

The source dependency recipe is retained, with upstream OpenPI licenses and source hashes. A clean GPU installation and retraining have not been repeated during packaging.

## Check, prepare, train

Use the canonical LeRobot layout `$LEROBOT_HOME/$REPO_ID` and a separate run directory for each task/method:

```bash
export LEROBOT_HOME=/path/to/lerobot
export REPO_ID=namespace/task_dataset
export BASE_PARAMS=/path/to/pi05_base/params
export RUN_DIR=/path/to/new/temporal_task_run
export METHOD=temporal_fusion  # or dual_path_fusion

"$TEMPORAL_PYTHON" run.py check --method "$METHOD" --lerobot-home "$LEROBOT_HOME" --repo-id "$REPO_ID" --run-dir "$RUN_DIR" --base-params "$BASE_PARAMS"
"$TEMPORAL_PYTHON" run.py stats --method "$METHOD" --lerobot-home "$LEROBOT_HOME" --repo-id "$REPO_ID" --run-dir "$RUN_DIR" --base-params "$BASE_PARAMS"
CUDA_VISIBLE_DEVICES=0 "$TEMPORAL_PYTHON" run.py train --method "$METHOD" --lerobot-home "$LEROBOT_HOME" --repo-id "$REPO_ID" --run-dir "$RUN_DIR" --base-params "$BASE_PARAMS"
```

`config` prints the resolved recipe. `check` decodes a real dataset sample and checks transformed shapes; it does not load model weights. `stats` refuses existing statistics. `train` refuses changed statistics or existing checkpoints.

## Serve

```bash
CUDA_VISIBLE_DEVICES=0 "$TEMPORAL_PYTHON" serve.py \
  --method "$METHOD" --lerobot-home "$LEROBOT_HOME" --repo-id "$REPO_ID" \
  --run-dir "$RUN_DIR" --base-params "$BASE_PARAMS" \
  --checkpoint "$RUN_DIR/checkpoints/forcebench_$METHOD/seed42/9999" --port 8000
```

The evaluator must maintain a 16-frame history at every 15 Hz control step and send `force_history[16,6]`, `force_history_is_pad[16]`, current `eef_wrench[6]`, RGB images, state, and prompt. Do not repeat only the current wrench or maintain history only at the 50-step policy-query rate. The portable server rejects a missing/incorrect history shape and returns absolute TCP actions `[50,7]`.
