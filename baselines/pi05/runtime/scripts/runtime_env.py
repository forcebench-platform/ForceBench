"""Set explicit writable paths before importing the model or dataset stack."""
import os
from pathlib import Path
import shutil
import sys

RUNTIME = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ["FORCEBENCH_RUN_ROOT"]).resolve()


def configure():
    sys.path.insert(0, str(RUNTIME / "src"))
    sys.dont_write_bytecode = True
    cache = ROOT / "cache"
    paths = {
        "HF_HOME": cache / "huggingface", "HF_DATASETS_CACHE": cache / "huggingface/datasets",
        "HF_HUB_CACHE": cache / "huggingface/hub", "OPENPI_DATA_HOME": cache / "openpi",
        "JAX_COMPILATION_CACHE_DIR": cache / "jax", "XDG_CACHE_HOME": cache / "xdg",
        "TMPDIR": ROOT / "tmp", "TEMP": ROOT / "tmp", "TMP": ROOT / "tmp",
        "WANDB_DIR": ROOT / "logs/wandb", "TORCH_HOME": cache / "torch",
        "CUDA_CACHE_PATH": cache / "cuda", "TRITON_CACHE_DIR": cache / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor", "MPLCONFIGDIR": cache / "matplotlib",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["HF_LEROBOT_HOME"] = os.environ["FORCEBENCH_LEROBOT_HOME"]
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.88")
    return ROOT


def require_space(path, gib=120):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free / 2**30
    if free < gib:
        raise OSError(f"Insufficient disk space: {free:.1f} GiB free; {gib} GiB required")
    return free
