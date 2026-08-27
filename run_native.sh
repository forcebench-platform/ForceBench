#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="${FORCEBENCH_RUNTIME_ROOT:-${root}/runtime}"
data_root="${FORCEBENCH_DATA_ROOT:-/home/pc/ForceBench_data}"
scratch_root="${FORCEBENCH_SCRATCH_ROOT:-/home/pc/ForceBench_scratch}"
isaaclab_root="${FORCEBENCH_ISAACLAB_ROOT:-${runtime_root}/IsaacLab}"
conda_sh="${FORCEBENCH_CONDA_SH:-${runtime_root}/miniforge/etc/profile.d/conda.sh}"
conda_env="${FORCEBENCH_CONDA_ENV:-env_isaacsim6}"

[[ -x "${isaaclab_root}/isaaclab.sh" ]] || { echo "Isaac Lab launcher missing: ${isaaclab_root}/isaaclab.sh" >&2; exit 1; }
[[ -f "${conda_sh}" ]] || { echo "Conda setup missing: ${conda_sh}" >&2; exit 1; }

# shellcheck disable=SC1090
source "${conda_sh}"
conda activate "${conda_env}"
export PYTHONPATH="${root}:${isaaclab_root}/source/isaaclab_visualizers${PYTHONPATH:+:${PYTHONPATH}}"
export FORCEBENCH_ASSET_DIR="${FORCEBENCH_ASSET_DIR:-${root}/assets}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
cd "${isaaclab_root}"

mode="${1:-collect}"
task="${FORCEBENCH_TASK:-peg_insert}"
policy_adapter_options="${FORCEBENCH_POLICY_ADAPTER_OPTIONS:-}"
if [[ -z "${policy_adapter_options}" ]]; then
  policy_adapter_options='{}'
fi
if [[ -z "${FORCEBENCH_SCENE_PROFILE:-}" ]]; then
  case "${task}" in
    rattle_sort) export FORCEBENCH_SCENE_PROFILE="parcel_sort_station" ;;
    weight_sort) export FORCEBENCH_SCENE_PROFILE="weight_sort_residential" ;;
    faucet_turn) export FORCEBENCH_SCENE_PROFILE="kitchen_sink" ;;
    stove_turn_on) export FORCEBENCH_SCENE_PROFILE="kitchen_stove" ;;
    *) export FORCEBENCH_SCENE_PROFILE="contact_lab" ;;
  esac
fi
case "${mode}" in
  expert)
    shift
    # Production scripted tasks are routed by collect_expert.py
    # to the production topology: one Isaac process with 32 cloned envs.
    exec python3 "${root}/scripts/expert/collect_expert.py" "$@"
    ;;
  collect)
    dataset="${2:-${data_root}/datasets/${task}}"
    dashboard_port="${3:-8765}"
    exec ./isaaclab.sh -p "${root}/scripts/collection/record_forcebench.py" \
      --task "${task}" --teleop_device gello --headless --enable_cameras --realtime \
      --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}" \
      --physics_rate_hz "${FORCEBENCH_PHYSICS_RATE_HZ:-60}" \
      --solver_position_iterations "${FORCEBENCH_SOLVER_POSITION_ITERATIONS:-64}" \
      --seed "${FORCEBENCH_SEED:-0}" \
      --dashboard_host 127.0.0.1 --dashboard_port "${dashboard_port}" \
      --dataset_dir "${dataset}"
    ;;
  smoke)
    dataset="${2:-${scratch_root}/smoke/${task}}"
    steps="${3:-3}"
    preview_args=()
    if [[ -n "${FORCEBENCH_PREVIEW_DIR:-}" ]]; then
      preview_args+=(--preview_dir "${FORCEBENCH_PREVIEW_DIR}")
    fi
    exec ./isaaclab.sh -p "${root}/scripts/collection/record_forcebench.py" \
      --task "${task}" --teleop_device zero --headless --enable_cameras \
      --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}" \
      --physics_rate_hz "${FORCEBENCH_PHYSICS_RATE_HZ:-60}" \
      --solver_position_iterations "${FORCEBENCH_SOLVER_POSITION_ITERATIONS:-64}" \
      --seed "${FORCEBENCH_SEED:-0}" --max_steps "${steps}" "${preview_args[@]}" \
      --dataset_dir "${dataset}"
    ;;
  grasp-smoke)
    exec ./isaaclab.sh -p "${root}/scripts/smoke/grasp_smoke.py" \
      --headless --enable_cameras --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}" \
      --physics_rate_hz "${FORCEBENCH_PHYSICS_RATE_HZ:-60}" \
      --solver_position_iterations "${FORCEBENCH_SOLVER_POSITION_ITERATIONS:-64}" \
      --seed "${FORCEBENCH_SEED:-42}"
    ;;
  drop-smoke)
    exec ./isaaclab.sh -p "${root}/scripts/smoke/drop_smoke.py" \
      --headless --enable_cameras --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}" \
      --physics_rate_hz "${FORCEBENCH_PHYSICS_RATE_HZ:-60}" \
      --solver_position_iterations "${FORCEBENCH_SOLVER_POSITION_ITERATIONS:-64}" \
      --seed "${FORCEBENCH_SEED:-0}"
    ;;
  faucet-detent-smoke)
    exec ./isaaclab.sh -p "${root}/scripts/smoke/faucet_detent_smoke.py" \
      --headless --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}" \
      --physics_rate_hz "${FORCEBENCH_PHYSICS_RATE_HZ:-60}" \
      --solver_position_iterations "${FORCEBENCH_SOLVER_POSITION_ITERATIONS:-64}"
    ;;
  faucet-contact-smoke)
    exec ./isaaclab.sh -p "${root}/scripts/smoke/faucet_contact_smoke.py" \
      --headless --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}" \
      --physics_rate_hz "${FORCEBENCH_PHYSICS_RATE_HZ:-60}" \
      --solver_position_iterations "${FORCEBENCH_SOLVER_POSITION_ITERATIONS:-64}"
    ;;
  faucet-views)
    output_dir="${2:-${scratch_root}/outputs/faucet_turn_views}"
    exec ./isaaclab.sh -p "${root}/scripts/smoke/capture_faucet_views.py" \
      --headless --enable_cameras --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}" \
      --output_dir "${output_dir}"
    ;;
  faucet-live-tune)
    tuning_output="${2:-${scratch_root}/outputs/faucet_live_tuning.json}"
    exec ./isaaclab.sh -p "${root}/scripts/tuning/faucet_static_tune.py" \
      --enable_cameras --viz kit --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}" \
      --seed "${FORCEBENCH_SEED:-0}" --output "${tuning_output}"
    ;;
  stove-detent-smoke)
    exec ./isaaclab.sh -p "${root}/scripts/smoke/stove_detent_smoke.py" \
      --headless --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}" \
      --physics_rate_hz "${FORCEBENCH_PHYSICS_RATE_HZ:-60}" \
      --solver_position_iterations "${FORCEBENCH_SOLVER_POSITION_ITERATIONS:-64}"
    ;;
  stove-views)
    output_dir="${2:-${scratch_root}/outputs/stove_turn_on_views}"
    exec ./isaaclab.sh -p "${root}/scripts/smoke/capture_stove_views.py" \
      --headless --enable_cameras --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}" \
      --output_dir "${output_dir}"
    ;;
  rattle-sort-views)
    output_dir="${2:-${scratch_root}/outputs/rattle_sort_views}"
    seed="${3:-1}"
    exec ./isaaclab.sh -p "${root}/scripts/smoke/capture_rattle_sort_views.py" \
      --output_dir "${output_dir}" --seed "${seed}" \
      --headless --enable_cameras --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}"
    ;;
  replay)
    episode="${2:?Usage: $0 replay EPISODE [PREVIEW_DIR]}"
    preview_dir="${3:-${scratch_root}/previews/replay}"
    exec ./isaaclab.sh -p "${root}/scripts/evaluation/replay_forcebench.py" "${episode}" \
      --headless --enable_cameras --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}" \
      --preview_dir "${preview_dir}"
    ;;
  policy)
    echo "Legacy single-environment policy evaluation was archived at:" >&2
    echo "  ${root}/archive/legacy_evaluation_20260825_061241" >&2
    echo "Use scripts/evaluation/evaluate_vectorized_seed_range.sh or '$0 policy-vector ...'." >&2
    exit 2
    ;;
  policy-vector)
    seed_start="${2:-${FORCEBENCH_SEED_START:-0}}"
    num_envs="${3:-${FORCEBENCH_NUM_ENVS:-10}}"
    max_steps="${4:-auto}"
    if [[ "${max_steps}" == "auto" ]]; then
      max_steps="$(
        PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}" \
          python3 "${root}/scripts/evaluation/resolve_task_max_steps.py" "${task}"
      )"
    fi
    [[ "${max_steps}" =~ ^[1-9][0-9]*$ ]] || {
      echo "resolved max_steps is not a positive integer: ${max_steps}" >&2
      exit 64
    }
    output_dir="${5:-${data_root}/evaluations/${task}/vector_${seed_start}_${num_envs}}"
    aggregate_filename="${6:-aggregate.json}"
    vector_termination_args=(--terminate_on_success)
    if [[ "${FORCEBENCH_TERMINATE_ON_SUCCESS:-1}" == "0" ]]; then
      vector_termination_args=(--no-terminate_on_success)
    fi
    exec ./isaaclab.sh -p "${root}/scripts/evaluation/inference_forcebench_vectorized.py" \
      --task "${task}" --enable_cameras --viz none \
      --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}" \
      --physics_rate_hz "${FORCEBENCH_PHYSICS_RATE_HZ:-60}" \
      --solver_position_iterations "${FORCEBENCH_SOLVER_POSITION_ITERATIONS:-64}" \
      --main_image_size "${FORCEBENCH_MAIN_IMAGE_SIZE:-224}" \
      --wrist_image_size "${FORCEBENCH_WRIST_IMAGE_SIZE:-224}" \
      --seed_start "${seed_start}" --num_envs "${num_envs}" \
      --env_spacing "${FORCEBENCH_ENV_SPACING:-6.0}" \
      --max_steps "${max_steps}" \
      --policy_host "${FORCEBENCH_POLICY_HOST:-127.0.0.1}" \
      --policy_port "${FORCEBENCH_POLICY_PORT:-8002}" \
      --policy_ports "${FORCEBENCH_POLICY_PORTS:-}" \
      --policy_adapter "${FORCEBENCH_POLICY_ADAPTER:-openpi_ws}" \
      --policy_adapter_options "${policy_adapter_options}" \
      --prompt "${FORCEBENCH_PROMPT:-}" \
      --execute_chunk "${FORCEBENCH_EXECUTE_CHUNK:-50}" \
      --success_stable_frames "${FORCEBENCH_SUCCESS_STABLE_FRAMES:-1}" \
      --final_settle_frames "${FORCEBENCH_FINAL_SETTLE_FRAMES:-8}" \
      --save_images --save_rollouts \
      --aggregate_filename "${aggregate_filename}" \
      "${vector_termination_args[@]}" \
      --output_dir "${output_dir}"
    ;;
  policy-replay)
    rollout_dir="${2:?Usage: $0 policy-replay ROLLOUT_DIR [VIDEO_PATH]}"
    video_path="${3:-${rollout_dir}/replay_main_wrist.mp4}"
    exec ./isaaclab.sh -p "${root}/scripts/evaluation/render_policy_rollout.py" \
      --rollout_dir "${rollout_dir}" --video_path "${video_path}" \
      --enable_cameras --viz none \
      --device "${FORCEBENCH_SIM_DEVICE:-cuda:0}" \
      --main_image_size "${FORCEBENCH_MAIN_IMAGE_SIZE:-224}" \
      --wrist_image_size "${FORCEBENCH_WRIST_IMAGE_SIZE:-224}" \
      --frame_stride "${FORCEBENCH_REPLAY_FRAME_STRIDE:-1}" \
      --physics_rate_hz "${FORCEBENCH_PHYSICS_RATE_HZ:-60}" \
      --solver_position_iterations "${FORCEBENCH_SOLVER_POSITION_ITERATIONS:-64}"
    ;;
  *)
    echo "Usage: $0 expert --list | $0 expert --task TASK --seeds SPEC [OPTIONS] (weight/rattle/stove/radmanso/peg default to 32 clones) | FORCEBENCH_TASK=<task> $0 {collect|smoke|policy-vector|replay} ..." >&2
    exit 2
    ;;
esac
