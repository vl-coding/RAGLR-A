from pathlib import Path

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def ensure_project_dirs(config: dict) -> None:
    paths = config.get("paths", {})

    for key in ("dense_index_dir", "logs_dir", "outputs_dir"):
        if key in paths:
            Path(paths[key]).mkdir(parents=True, exist_ok=True)

    for key in ("bm25_index", "keyword_index", "metadata_db", "manifest", "update_state"):
        if key in paths:
            Path(paths[key]).parent.mkdir(parents=True, exist_ok=True)

    data = config.get("data", {})
    for key in ("processed_path",):
        if key in data:
            Path(data[key]).parent.mkdir(parents=True, exist_ok=True)
