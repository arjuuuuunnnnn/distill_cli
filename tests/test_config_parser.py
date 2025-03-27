import pytest
import json
import yaml
from pathlib import Path
from distill_cli.cli import load_config

@pytest.fixture
def json_config_file(tmp_path):
    config = {
        "teacher": {"name": "bert-base-uncased", "type": "huggingface", "from_pretrained": True},
        "student": {"name": "distilbert-base-uncased", "type": "huggingface", "from_pretrained": False},
        "data": {"type": "huggingface", "name": "imdb"},
        "framework": "pytorch",
        "training": {"epochs": 3, "batch_size": 16}
    }
    file_path = tmp_path / "config.json"
    with open(file_path, "w") as f:
        json.dump(config, f)
    return file_path

@pytest.fixture
def yaml_config_file(tmp_path):
    config = {
        "teacher": {"name": "bert-base-uncased", "type": "huggingface", "from_pretrained": True},
        "student": {"name": "distilbert-base-uncased", "type": "huggingface", "from_pretrained": False},
        "data": {"type": "huggingface", "name": "imdb"},
        "framework": "pytorch",
        "training": {"epochs": 3, "batch_size": 16}
    }
    file_path = tmp_path / "config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config, f)
    return file_path

def test_load_json_config(json_config_file):
    config = load_config(json_config_file)
    assert config["teacher"]["name"] == "bert-base-uncased"
    assert config["framework"] == "pytorch"
    assert config["training"]["epochs"] == 3

def test_load_yaml_config(yaml_config_file):
    config = load_config(yaml_config_file)
    assert config["teacher"]["name"] == "bert-base-uncased"
    assert config["framework"] == "pytorch"
    assert config["training"]["epochs"] == 3

def test_invalid_config_format(tmp_path):
    invalid_file = tmp_path / "config.txt"
    invalid_file.write_text("invalid content")
    with pytest.raises(ValueError, match="Unsupported config format"):
        load_config(invalid_file)

def test_missing_config_file():
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config("non_existent_file.yaml")
