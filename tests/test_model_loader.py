import pytest
from distill_cli.core.model_loader import ModelLoader
from transformers import BertForSequenceClassification, TFBertForSequenceClassification

@pytest.fixture
def mock_pytorch_config():
    return {
        "name": "bert-base-uncased",
        "type": "huggingface",
        "from_pretrained": True,
        "task": "sequence-classification"
    }

@pytest.fixture
def mock_tf_config():
    return {
        "name": "bert-base-uncased",
        "type": "huggingface",
        "from_pretrained": True,
        "task": "sequence-classification",
        "framework": "tensorflow"
    }

def test_load_pytorch_model(mock_pytorch_config):
    model = ModelLoader.load_model(mock_pytorch_config)
    assert isinstance(model, BertForSequenceClassification)

def test_load_tf_model(mock_tf_config):
    model = ModelLoader.load_model(mock_tf_config)
    assert isinstance(model, TFBertForSequenceClassification)
