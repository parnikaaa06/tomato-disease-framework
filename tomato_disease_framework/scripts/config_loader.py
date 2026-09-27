import os
from pathlib import Path
import yaml


def load_config(config_path="config.yaml"):
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    raw_text = config_path.read_text(encoding="utf-8")

    # Expand environment variables such as ${TDF_DATASET_ROOT}
    expanded_text = os.path.expandvars(raw_text)

    config = yaml.safe_load(expanded_text)

    if not isinstance(config, dict):
        raise ValueError("config.yaml did not produce a valid dictionary.")

    return config


def get_dataset_root(config):
    dataset_root = config["data"]["external_dataset_path"]

    # Environment variable is not configured
    if not dataset_root or dataset_root.startswith("${"):
        return None

    return Path(dataset_root)
