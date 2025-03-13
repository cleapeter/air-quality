import pytest

from config import load_config


def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent.yaml")


def test_load_config_valid_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_content = """
    data:
      target_column: "CO_reference_mg_per_m3"
    """
    config_path.write_text(config_content)

    config = load_config(str(config_path))
    assert "data" in config
    assert config["data"]["target_column"] == "CO_reference_mg_per_m3"
