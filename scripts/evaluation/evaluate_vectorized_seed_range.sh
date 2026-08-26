#!/usr/bin/env bash
set -uo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
task="${1:?usage: $0 TASK FIRST_SEED LAST_SEED MAX_STEPS_OR_AUTO OUTPUT_DIR [NUM_ENVS=auto]}"
first_seed="${2:?missing first seed}"
last_seed="${3:?missing last seed}"
max_steps="${4:?missing max steps; use auto for the task-configured limit}"
output_dir="${5:?missing output directory}"
num_envs="${6:-auto}"
host="${CORIBENCH_POLICY_HOST:?set CORIBENCH_POLICY_HOST}"
port="${CORIBENCH_POLICY_PORT:-8002}"
ports="${CORIBENCH_POLICY_PORTS:-}"
adapter="${CORIBENCH_POLICY_ADAPTER:?set CORIBENCH_POLICY_ADAPTER}"
env_spacing="${CORIBENCH_ENV_SPACING:-6.0}"
adapter_options="${CORIBENCH_POLICY_ADAPTER_OPTIONS:-}"
if [[ -z "$adapter_options" ]]; then
  adapter_options='{}'
fi

[[ "$first_seed" =~ ^[0-9]+$ && "$last_seed" =~ ^[0-9]+$ ]] || exit 64
(( last_seed >= first_seed )) || exit 64
if [[ "$max_steps" == "auto" ]]; then
  max_steps="$(
    PYTHONPATH="$root${PYTHONPATH:+:${PYTHONPATH}}" \
      python3 "$root/scripts/evaluation/resolve_task_max_steps.py" "$task"
  )" || exit 64
  max_steps_source=task_manifest
else
  max_steps_source=explicit_override
fi
[[ "$max_steps" =~ ^[1-9][0-9]*$ ]] || {
  echo "resolved max_steps is not a positive integer: $max_steps" >&2
  exit 64
}
if [[ "$num_envs" == "auto" ]]; then
  num_envs="$(
    PYTHONPATH="$root${PYTHONPATH:+:${PYTHONPATH}}" \
      python3 "$root/scripts/evaluation/resolve_vector_batch_size.py" "$task"
  )" || exit 64
  batch_size_source=scene_profile
else
  batch_size_source=explicit_override
fi
[[ "$num_envs" =~ ^[1-9][0-9]*$ ]] || {
  echo "resolved num_envs is not a positive integer: $num_envs" >&2
  exit 64
}
(( num_envs <= 100 )) || {
  echo "num_envs above 100 has not been validated on the L20" >&2
  exit 64
}
printf 'EVAL_BATCH_SIZE task=%s num_envs=%s source=%s\n' \
  "$task" "$num_envs" "$batch_size_source"
printf 'EVAL_MAX_STEPS task=%s max_steps=%s source=%s\n' \
  "$task" "$max_steps" "$max_steps_source"
if test -e "$output_dir"; then
  echo "evaluation output already exists: $output_dir" >&2
  exit 73
fi
mkdir -p "$output_dir"

run_batch() {
  local batch_start=$1
  local batch_count=$2
  local batch_end=$((batch_start + batch_count - 1))
  local batch_name
  local log
  local status_file
  local aggregate_filename
  local started
  local status
  batch_name="batch_$(printf '%03d' "$batch_start")_$(printf '%03d' "$batch_end")"
  log="$output_dir/${batch_name}.log"
  status_file="$output_dir/${batch_name}.status"
  aggregate_filename="${batch_name}.aggregate.json"
  started=$(date +%s)
  if CORIBENCH_TASK="$task" \
    CORIBENCH_POLICY_HOST="$host" \
    CORIBENCH_POLICY_PORT="$port" \
    CORIBENCH_POLICY_PORTS="$ports" \
    CORIBENCH_POLICY_ADAPTER="$adapter" \
    CORIBENCH_ENV_SPACING="$env_spacing" \
    CORIBENCH_GUI=0 \
    CORIBENCH_EXECUTE_CHUNK="${CORIBENCH_EXECUTE_CHUNK:-50}" \
    CORIBENCH_POLICY_ADAPTER_OPTIONS="$adapter_options" \
      "$root/run_native.sh" policy-vector \
        "$batch_start" "$batch_count" "$max_steps" "$output_dir" \
        "$aggregate_filename" >"$log" 2>&1; then
    status=0
  else
    status=$?
  fi
  printf '%s\t%s\n' "$status" "$(( $(date +%s) - started ))" >"$status_file"
  return 0
}

batch_start=$first_seed
batch_count_total=0
while (( batch_start <= last_seed )); do
  remaining=$((last_seed - batch_start + 1))
  batch_count=$num_envs
  if (( remaining < batch_count )); then
    batch_count=$remaining
  fi
  run_batch "$batch_start" "$batch_count"
  batch_start=$((batch_start + batch_count))
  batch_count_total=$((batch_count_total + 1))
done

python3 - "$output_dir" "$task" "$first_seed" "$last_seed" "$max_steps" \
  "$num_envs" "$host" "$port" "$adapter" \
  "${CORIBENCH_MODEL_NAME:-unknown}" "${CORIBENCH_CHECKPOINT:-unknown}" \
  "$batch_count_total" "$ports" <<'PY'
import json
from pathlib import Path
import re
import statistics
import sys

out = Path(sys.argv[1])
task = sys.argv[2]
first, last, max_steps, num_envs = map(int, sys.argv[3:7])
batch_pattern = re.compile(r"batch_(\d+)_(\d+)\.status$")
batches = []
for status_path in sorted(out.glob("batch_*.status")):
    match = batch_pattern.fullmatch(status_path.name)
    if match is None:
        continue
    values = status_path.read_text(encoding="utf-8").strip().split("\t")
    process_status, elapsed = map(int, values)
    batches.append({
        "seed_start": int(match.group(1)),
        "seed_end": int(match.group(2)),
        "process_status": process_status,
        "elapsed_s": elapsed,
    })

records = []
for seed in range(first, last + 1):
    summary_path = out / f"seed_{seed:03d}" / "summary.json"
    batch = next(
        (item for item in batches if item["seed_start"] <= seed <= item["seed_end"]),
        None,
    )
    if not summary_path.is_file():
        records.append({
            "seed": seed,
            "completed": False,
            "success": False,
            "process_status": None if batch is None else batch["process_status"],
            "elapsed_s": None if batch is None else batch["elapsed_s"],
        })
        continue
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records.append({
        "seed": seed,
        "completed": True,
        "success": bool(summary.get("success", False)),
        "process_status": None if batch is None else batch["process_status"],
        "elapsed_s": None if batch is None else batch["elapsed_s"],
        "steps": int(summary.get("steps", 0)),
        "mean_inference_latency_s": summary.get("mean_inference_latency_s"),
        "termination_reason": summary.get("termination_reason"),
        "success_report": summary.get("success_report", {}),
    })

completed = [record for record in records if record["completed"]]
successes = [record for record in completed if record["success"]]
latencies = [
    record["mean_inference_latency_s"]
    for record in completed
    if record.get("mean_inference_latency_s") is not None
]
result = {
    "schema": "coribench_vectorized_policy_evaluation_v1",
    "task": task,
    "seed_range": [first, last],
    "requested_rollouts": last - first + 1,
    "completed_rollouts": len(completed),
    "successful_rollouts": len(successes),
    "success_rate": len(successes) / len(completed) if completed else None,
    "num_envs_per_batch": num_envs,
    "batch_count": int(sys.argv[12]),
    "max_steps": max_steps,
    "policy_host": sys.argv[7],
    "policy_port": int(sys.argv[8]),
    "policy_ports": [
        int(value) for value in sys.argv[13].split(",") if value
    ] or [int(sys.argv[8])],
    "policy_adapter": sys.argv[9],
    "model_name": sys.argv[10],
    "checkpoint": sys.argv[11],
    "mean_inference_latency_s": statistics.fmean(latencies) if latencies else None,
    "batches": batches,
    "records": records,
}
(out / "aggregate.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
)
print("CORIBENCH_VECTOR_EVAL_COMPLETE=" + json.dumps({
    key: result[key]
    for key in (
        "task",
        "requested_rollouts",
        "completed_rollouts",
        "successful_rollouts",
        "success_rate",
        "num_envs_per_batch",
        "batch_count",
    )
}, separators=(",", ":")))
if len(completed) != result["requested_rollouts"]:
    raise SystemExit(1)
PY
