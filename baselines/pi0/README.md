# pi0 reference policy

This is the observed September 15 pi0 adapter and OpenPI source, packaged with explicit local paths. It uses **pi0**, not pi0.5: pi0 base weights, continuous state input, 10,000 updates, global batch 32, horizon/execution 50, seed 42, and task-specific normalization. Force is not an input to this reference.

The adapter's data and model transforms are unchanged. Private launchers and the upstream example training registry were omitted. The original upstream commit was `215abfb217dbac7d5f1273282331b9b1866c0479`. `SOURCE_MANIFEST.json` records copied source hashes before packaging changes. Upstream licenses are in `openpi/`.

## Environment

Use Linux, Python 3.11, and a separate environment from the other OpenPI variants:

```bash
cd baselines/pi0/openpi
uv sync --python 3.11
cd ..
export PI0_PYTHON="$PWD/openpi/.venv/bin/python"
```

`pyproject.toml` retains the observed JAX/Flax/LeRobot recipe. GPU support depends on the installed CUDA runtime/driver. This release has not repeated a clean GPU installation or retrained the models.

## Train

The dataset must already contain 15 Hz `image`, `wrist_image`, `state[7]`, and `actions[7]` fields. Select the official task dataset from the release dataset manifest; do not substitute a same-named older snapshot.

```bash
# Run from baselines/pi0. Set these to your own data and downloaded pi0_base.
export DATASET=/path/to/lerobot/namespace/task_dataset
export REPO_ID=namespace/task_dataset
export RUN_DIR=/path/to/new/pi0_task_run
export BASE_PARAMS=/path/to/pi0_base/params

bash pi0_coribench/run_pi0.sh check --dataset "$DATASET" --repo-id "$REPO_ID" --run-dir "$RUN_DIR" --base-params "$BASE_PARAMS"
bash pi0_coribench/run_pi0.sh stats --dataset "$DATASET" --repo-id "$REPO_ID" --run-dir "$RUN_DIR" --base-params "$BASE_PARAMS"
CUDA_VISIBLE_DEVICES=0 bash pi0_coribench/run_pi0.sh train --dataset "$DATASET" --repo-id "$REPO_ID" --run-dir "$RUN_DIR" --base-params "$BASE_PARAMS"
```

The stats command computes the same wrapped TCP delta-action statistics used for training. The training command checks dataset/config/source identity against that stats run and refuses an existing checkpoint output. Final checkpoint numbering is `9999` after 10,000 updates.

## Serve

```bash
CUDA_VISIBLE_DEVICES=0 "$PI0_PYTHON" serve.py \
  --repo-id "$REPO_ID" --run-dir "$RUN_DIR" --base-params "$BASE_PARAMS" \
  --checkpoint "$RUN_DIR/checkpoints/coribench_pi0_10k/seed42/9999" --port 8000
```

The websocket adapter takes `observation/state`, `observation/image`, `observation/wrist_image`, and `prompt`, and returns absolute TCP xyz/rpy/gripper actions `[50,7]`. The evaluator must execute 50 actions per query. Serving loads normalization from the checkpoint. The default bind address is loopback; `--host` is explicit for remote deployment.
