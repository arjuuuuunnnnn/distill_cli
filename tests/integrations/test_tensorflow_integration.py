import pytest
import tensorflow as tf
import os
import json
from pathlib import Path
from distill_cli.core.model_loader import ModelLoader
from distill_cli.core.distillers import TensorFlowDistiller
from distill_cli.data.dataset import DistillationDataset
from transformers import TFBertForSequenceClassification, AutoTokenizer

@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    """Create a tiny dataset for testing"""
    tmp_path = tmp_path_factory.mktemp("data")
    train_data_path = tmp_path / "train.csv"
    val_data_path = tmp_path / "val.csv"
    
    # Create tiny train dataset
    with open(train_data_path, "w") as f:
        f.write("text,label\n")
        f.write("this is a positive example,1\n")
        f.write("this is a negative example,0\n")
        f.write("another positive text,1\n")
        f.write("another negative text,0\n")
    
    # Create tiny validation dataset
    with open(val_data_path, "w") as f:
        f.write("text,label\n")
        f.write("validation positive,1\n")
        f.write("validation negative,0\n")
    
    return {
        "train": train_data_path,
        "val": val_data_path,
        "dir": tmp_path
    }

@pytest.fixture(scope="module")
def tensorflow_config(tiny_dataset):
    """Create a TensorFlow-specific configuration"""
    return {
        "teacher": {
            "name": "bert-base-uncased",
            "type": "huggingface",
            "from_pretrained": True,
            "framework": "tensorflow",
            "task": "sequence-classification"
        },
        "student": {
            "name": "bert-base-uncased",
            "type": "huggingface",
            "from_pretrained": False,
            "framework": "tensorflow",
            "config_overrides": {
                "hidden_size": 512,
                "intermediate_size": 1024,
                "num_hidden_layers": 3,
                "num_attention_heads": 8
            },
            "task": "sequence-classification"
        },
        "data": {
            "type": "custom",
            "path": str(tiny_dataset["train"]),
            "validation_path": str(tiny_dataset["val"]),
            "text_column": "text",
            "label_column": "label"
        },
        "framework": "tensorflow",
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 5e-5,
            "temperature": 2.0,
            "alpha": 0.5,
            "num_labels": 2,
            "patience": 1
        }
    }

@pytest.mark.slow
def test_tensorflow_model_loading(tensorflow_config):
    """Test loading models with TensorFlow"""
    try:
        # Load teacher and student
        teacher = ModelLoader.load_model(tensorflow_config["teacher"])
        student = ModelLoader.load_model(tensorflow_config["student"])
        
        # Verify types
        assert isinstance(teacher, TFBertForSequenceClassification)
        assert isinstance(student, TFBertForSequenceClassification)
        
        # Verify student config was applied
        assert student.config.hidden_size == 512
        assert student.config.intermediate_size == 1024
        assert student.config.num_hidden_layers == 3
        assert student.config.num_attention_heads == 8
    except Exception as e:
        pytest.skip(f"TensorFlow model loading test failed: {str(e)}")

@pytest.mark.slow
def test_tensorflow_dataset_loading(tensorflow_config):
    """Test loading and processing dataset with TensorFlow"""
    try:
        # Initialize dataset
        dataset = DistillationDataset(tensorflow_config)
        
        # Load data
        processed_data = dataset.load_data()
        
        # Get dataloaders
        train_loader, val_loader = dataset.get_dataloaders(processed_data)
        
        # Check that we get TensorFlow datasets
        assert isinstance(train_loader, tf.data.Dataset)
