import logging

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    try:
        logger.info(f"Loading configuration from {config_path}...")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check if 'target_column' exists in the configuration
        if "target_column" not in config["data"]:
            raise KeyError("Configuration error: 'target_column' not found in config['data']")
        target_column = config["data"]["target_column"]

        # Check if 'model_choice' exists in the configuration
        if "model_choice" not in config["data"]:
            raise KeyError("Configuration error: 'model_choice' not found in config['data']")
        model_name = config["data"]["model_choice"]

        # Check if the selected model exists in the models section
        if model_name not in config["models"]:
            raise KeyError(f"Configuration error: '{model_name}' not found in the models section")

        logger.info(f"Configuration loaded successfully. Target column: {target_column}. Using model: {model_name}")

        return config, target_column, model_name

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise
