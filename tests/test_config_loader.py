import copy
from unittest.mock import mock_open, patch

import pytest
import yaml

from config_loader import load_config

valid_config = {
    "data": {"target_column": "column1", "model_choice": "modelA"},
    "models": {"modelA": {"param1": "value1"}, "modelB": {"param2": "value2"}},
}


def test_load_config_success():
    with patch("builtins.open", mock_open(read_data=yaml.dump(valid_config))):
        config, target_column, model_name = load_config("config.yaml")

        assert config["data"]["target_column"] == "column1"
        assert config["data"]["model_choice"] == "modelA"
        assert model_name == "modelA"
        assert target_column == "column1"


def test_missing_target_column():
    invalid_config = copy.deepcopy(valid_config)
    del invalid_config["data"]["target_column"]

    with patch("builtins.open", mock_open(read_data=yaml.dump(invalid_config))):
        with pytest.raises(KeyError, match=r"Configuration error: 'target_column' not found in config\['data'\]"):
            load_config("config.yaml")


def test_missing_model_choice():
    invalid_config = copy.deepcopy(valid_config)
    del invalid_config["data"]["model_choice"]

    with patch("builtins.open", mock_open(read_data=yaml.dump(invalid_config))):
        with pytest.raises(KeyError, match=r"Configuration error: 'model_choice' not found in config\['data'\]"):
            load_config("config.yaml")


def test_missing_model_in_models():
    invalid_config = copy.deepcopy(valid_config)
    model_choice = invalid_config["data"]["model_choice"]
    del invalid_config["models"][model_choice]

    with patch("builtins.open", mock_open(read_data=yaml.dump(invalid_config))):
        with pytest.raises(KeyError, match=f"Configuration error: '{model_choice}' not found in the models section"):
            load_config("config.yaml")


def test_file_not_found():
    with patch("builtins.open", side_effect=FileNotFoundError("No such file or directory")):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


def test_empty_yaml():
    empty_yaml = ""
    with patch("builtins.open", mock_open(read_data=empty_yaml)):
        with pytest.raises(TypeError):
            load_config("config.yaml")
