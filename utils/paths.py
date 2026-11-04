"""本地路径配置；环境变量可把大型输入和新输出放到其他磁盘。"""

import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT
DATA_DIR = (
    Path(os.environ.get("BOUNDARY_DATA_DIR", str(PROJECT_ROOT / "data")))
    .expanduser()
    .resolve()
)
DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = (
    Path(os.environ.get("BOUNDARY_RUNS_DIR", str(DATA_DIR / "training_runs")))
    .expanduser()
    .resolve()
)
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = (
    Path(os.environ.get("BOUNDARY_RESULTS_DIR", str(PROJECT_ROOT / "outputs" / "results")))
    .expanduser()
    .resolve()
)
FIGURES_DIR = (
    Path(
        os.environ.get(
            "BOUNDARY_FIGURES_DIR", str(PROJECT_ROOT / "outputs" / "figures")
        )
    )
    .expanduser()
    .resolve()
)
LOGS_DIR = (
    Path(os.environ.get("BOUNDARY_LOGS_DIR", str(PROJECT_ROOT / "outputs" / "logs")))
    .expanduser()
    .resolve()
)


_RESULT_PATHS = json.loads((PROJECT_ROOT / "configs/result_paths.json").read_text())


def result_file(relative):
    path = RESULTS_DIR / _RESULT_PATHS.get(str(relative), str(relative))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_foundation_checkpoint(model_id):
    """Use a local model directory or download the named ModelScope checkpoint."""
    local = MODELS_DIR / model_id.replace("/", "__")
    if (local / "config.json").is_file():
        return str(local)
    from modelscope.hub.snapshot_download import snapshot_download
    return snapshot_download(model_id, cache_dir=str(MODELS_DIR))
