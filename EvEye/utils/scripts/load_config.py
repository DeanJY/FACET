from typing import Any
from pathlib import Path
import json
import yaml


def load_config(
    config_file_name: str, config_dir_path: Path = Path("configs")
) -> dict[str, Any]:
    config_path = config_dir_path / config_file_name
    with config_path.open() as config_file:
        if str(config_path).endswith("yaml"):
            config = yaml.safe_load(config_file)
        else:
            config = json.load(config_file)
    assert isinstance(
        config, dict
    ), f"Config file {str(config_path)} is not a dictionary"
    return config
