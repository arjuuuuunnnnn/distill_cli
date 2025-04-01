import pytest
from distill_cli.data.dataset import DistillationDataset
from transformers import AutoTokenizer
from datasets import Dataset

@pytest.fixture
def mock_config():
    return {
        "teacher": {"name": "bert-base-uncased"},
        "data": {
            "type": "huggingface",
            "name": "imdb",
            "text_column": "text",
            "label_column": "label"
        },
        "training": {"batch_size": 16},
        "framework": "pytorch"
    }

def test_load_hf_dataset(mock_config):
    dataset = DistillationDataset(mock_config)
    data = dataset.load_data()
    assert "train" in data and "test" in data

def test_process_data(mock_config):
    dataset = DistillationDataset(mock_config)
    tokenizer = AutoTokenizer.from_pretrained(mock_config["teacher"]["name"])
    mock_data = Dataset.from_dict({"text": ["sample text"], "label": [1]})
    processed_data = dataset._process_data(mock_data)
    assert "input_ids" in processed_data.column_names
    assert "attention_mask" in processed_data.column_names
    assert "labels" in processed_data.column_names

def test_get_dataloaders(mock_config):
    dataset = DistillationDataset(mock_config)
    mock_data = Dataset.from_dict({"text": ["sample text"] * 100, "label": [1] * 100})
    processed_data = dataset._process_data(mock_data)
    train_loader, val_loader = dataset.get_dataloaders({"train": processed_data, "test": processed_data})
    assert len(train_loader) > 0
    assert len(val_loader) > 0
