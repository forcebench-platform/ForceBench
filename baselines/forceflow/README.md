# ForceFlow reference adapter

This package contains the actual ForceBench adapter used in the September 15 formal runs, plus the required CleanDiffuser Python source. ForceFlow upstream: `544bb35c8363521c19b7425c529364ae4b80032d`; CleanDiffuser: `b6df692788cd03127c6dcfc515c2638444e71e9d`. Their licenses are preserved. `SOURCE_MANIFEST.json` records hashes of the copied source.

This is a ForceBench data/interface adaptation of ForceFlow, not a claim that every upstream experiment was reproduced. The observed recipe is 100,000 updates, global batch 128, Adam at 3e-4, no weight decay/scheduler/gradient clipping, two image observations, ten 15 Hz wrench samples, 64 predicted action steps, and 32 executed steps. Inference uses 20 ODE steps and no EMA.

## Environment

Use a separate Python 3.11 environment on Linux:

```bash
cd baselines/forceflow
python3.11 -m venv .venv
source .venv/bin/activate
# Choose the PyTorch CUDA build for your machine before installing requirements.
python -m pip install -r requirements.txt
```

CleanDiffuser is imported from the bundled directory; do not install its broad upstream environment requirements for unrelated benchmarks. Version pins reflect the observed B200 environment. The formal PyAV decoder is included and does not require LeRobot; alternative decoder backends use an optional installed LeRobot implementation. The GPU execution check below used an existing Python 3.11 / PyTorch 2.7.1+cu126 H200 environment; a fresh environment installation has not been verified.

## Configure and train

```bash
export DATASET=/path/to/lerobot/namespace/task_dataset
export RUN_DIR=/path/to/new/forceflow_task_run
python -m coribench_forceflow.prepare_config --task nut_thread \
  --dataset-root "$DATASET" --output-dir "$RUN_DIR" --config task.yaml --devices 1
python -m coribench_forceflow.train --config task.yaml --print-effective-config
CUDA_VISIBLE_DEVICES=0 python -m coribench_forceflow.train --config task.yaml
```

Choose one of the eight paper task IDs. The generator keeps global batch 128 when `--devices` changes (for example two devices, 64 samples each). It does not create or resample data. The formal Nut Thread, Stove Turn, and Stove Press snapshots used two GPUs; generated single-GPU configuration preserves the global batch but is a deployment change.

## Serve

```bash
CUDA_VISIBLE_DEVICES=0 python -m coribench_forceflow.server \
  --config task.yaml --checkpoint /path/to/step-100000.ckpt --port 8000
```

Pass the actual checkpoint filename from the run. The evaluator must collect dense observation history at **every 15 Hz control step** and send the latest window with each query, once per 32 executed actions. Include a unique `rollout_id`. A sequence of sparse policy-query observations is not a valid ten-frame force history.

Select `--policy_adapter forceflow_ws` in the vector evaluator and use `FORCEBENCH_IMAGE_TRANSPORT=raw` (the default). This adapter translates the canonical request's wire name to the preserved server's `coribench_policy`, supplies two image/state frames and ten wrench frames, and requires exactly 32 returned absolute TCP actions. JPEG transport is rejected before connecting because the preserved server cannot decode it. The numerical server remains unchanged. The plain `forcebench_ws` adapter uses a different wire name and cannot connect to this archived server.

The 64-predicted/32-executed contract is checked in CPU integration tests against the server's actual history parser and output conversion plus the evaluator's queueing block. The separate GPU check below also exercises the real model and native WebSocket server. A short execution check does not reproduce published task success rates.

Packaging changes: portable launcher; eight-task config choices; explicit GPU count; pinned upstream provenance instead of querying the enclosing ForceBench Git commit; lazy optional LeRobot import outside the formal PyAV decoder path. The numerical model, dataset transforms, and server inference code are retained.

## GPU execution validation

On 2026-10-21, the candidate source passed a real H200 check using the preserved final Weight Sort checkpoint and its paired normalization statistics. All 200 recorded episodes (91,368 frames) were loaded; the six position/action/force min/max arrays matched the checkpoint normalization exactly. Native inference used real recorded images, state and ten contiguous wrench frames with 20 ODE steps. The actual `forceflow_ws` adapter and MessagePack WebSocket server returned the same finite `[32, 7]` action array as direct inference, with zero numerical difference.

Two small-batch optimization steps then produced finite, nonzero gradients and changed model parameters. The newly saved validation checkpoint loaded strictly and ran inference. LeRobot imports were deliberately blocked throughout this final check. The original input checkpoint remained unchanged. A separate run of the actual `python -m coribench_forceflow.train` entry point also passed two updates from a fresh initialization on the real dataset, wrote finite CSV loss metrics, and saved a checkpoint with `global_step=2`. This validates execution on Weight Sort; it does not establish full training convergence or reproduce the eight-task paper scores. See the [GPU validation receipt](../../validation/public_report.json).

To repeat the check with an authentic checkpoint/config/dataset pair, install the ForceBench client dependencies in the same separate environment:

```bash
cd baselines/forceflow
python -m pip install -r requirements.txt -e "../..[dev]"
CUDA_VISIBLE_DEVICES=0 FORCEBENCH_IMAGE_TRANSPORT=raw python validate_gpu.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --normalizer "$NORMALIZER" \
  --dataset-root "$DATASET" --output "$NEW_VALIDATION_DIR" --train-steps 2
```

`NEW_VALIDATION_DIR` must not exist. The helper writes predictions, a receipt and a new validation checkpoint there. It reads the supplied dataset and original checkpoint without modifying them, checks the full dataset's normalization against the supplied statistics, and fails on mismatches. Batch size two and two optimization steps are intentional execution-test settings; the full training recipe remains global batch 128 and 100,000 updates.
