import os
from typing import Dict, Any
import yaml

def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """
    Safely loads and returns a dictionary from a YAML configuration file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config or {}

def load_all_configs(configs_dir: str) -> Dict[str, Any]:
    """
    Loads all YAML configuration files from configs_dir into a single dictionary.
    """
    merged_config = {}
    config_files = ["config.yaml", "dataset.yaml", "federated.yaml", "continual.yaml", "experiments.yaml"]
    for filename in config_files:
        full_path = os.path.join(configs_dir, filename)
        if os.path.exists(full_path):
            key_name = os.path.splitext(filename)[0]
            merged_config[key_name] = load_yaml_config(full_path)
    return merged_config
