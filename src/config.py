import yaml
import os


def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
