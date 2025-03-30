import logging

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    try:
        logger.info(f"Loading configuration from {config_path}...")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if "target_column" not in config["data"]:
            raise KeyError("Configuration error: 'target_column' not found in config['data']")

        target_column = config["data"]["target_column"]

        logger.info(f"Configuration loaded successfully. Target column: {target_column}")

        return config, target_column

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise
