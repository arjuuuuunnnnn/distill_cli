import importlib
import os
from pathlib import Path
from typing import Union, Dict, Any
import torch
import tensorflow as tf
from transformers import (
    AutoModel, 
    AutoModelForSequenceClassification,
    AutoTokenizer, 
    TFAutoModel,
    TFAutoModelForSequenceClassification,
    AutoConfig
)

class ModelLoader:
    """Universal model loader supporting PyTorch, TensorFlow, and HuggingFace models"""
    
    @staticmethod
    def load_model(model_config: Dict[str, Any]) -> Union[torch.nn.Module, tf.keras.Model]:
        model_type = model_config.get('type', 'pytorch')
        model_name = model_config['name']
        from_pretrained = model_config.get('from_pretrained', True)
        custom_weights = model_config.get('weights')
        model_class = model_config.get('class_name')
        task = model_config.get('task', 'sequence-classification')
        
        if model_type == 'pytorch':
            return ModelLoader._load_pytorch_model(model_name, from_pretrained, custom_weights, model_class, task)
        elif model_type == 'tensorflow':
            return ModelLoader._load_tf_model(model_name, from_pretrained, custom_weights, model_class, task)
        elif model_type == 'huggingface':
            return ModelLoader._load_hf_model(model_name, from_pretrained, model_config.get('framework', 'pytorch'), task)
        elif model_type == 'custom':
            return ModelLoader._load_custom_model(model_name, custom_weights, model_class)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def _load_pytorch_model(name: str, from_pretrained: bool, weights: str, class_name: str = None, task: str = 'sequence-classification'):
        if class_name:
            return ModelLoader._load_custom_model(name, weights, class_name)
            
        if from_pretrained:
            if task == 'sequence-classification':
                model = AutoModelForSequenceClassification.from_pretrained(name)
            else:
                model = AutoModel.from_pretrained(name)
        else:
            config = AutoConfig.from_pretrained(name)
            if task == 'sequence-classification':
                model = AutoModelForSequenceClassification.from_config(config)
            else:
                model = AutoModel.from_config(config)
        
        if weights:
            state_dict = torch.load(weights)
            model.load_state_dict(state_dict)
        return model

    @staticmethod
    def _load_tf_model(name: str, from_pretrained: bool, weights: str, class_name: str = None, task: str = 'sequence-classification'):
        if class_name:
            return ModelLoader._load_custom_model(name, weights, class_name)
            
        if from_pretrained:
            if task == 'sequence-classification':
                model = TFAutoModelForSequenceClassification.from_pretrained(name)
            else:
                model = TFAutoModel.from_pretrained(name)
        else:
            config = AutoConfig.from_pretrained(name)
            if task == 'sequence-classification':
                model = TFAutoModelForSequenceClassification.from_config(config)
            else:
                model = TFAutoModel.from_config(config)
        
        if weights:
            model.load_weights(weights)
        return model

    @staticmethod
    def _load_hf_model(name: str, from_pretrained: bool, framework: str, task: str = 'sequence-classification', config_overrides=None):
        if from_pretrained:
            if task == 'sequence-classification':
                if framework == 'tensorflow':
                    return TFAutoModelForSequenceClassification.from_pretrained(name)
                else:
                    return AutoModelForSequenceClassification.from_pretrained(name)
            else:
                if framework == 'tensorflow':
                    return TFAutoModel.from_pretrained(name)
                else:
                    return AutoModel.from_pretrained(name)
        
        config = AutoConfig.from_pretrained(name)
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)

        if task == 'sequence-classification':
            if framework == 'tensorflow':
                return TFAutoModelForSequenceClassification.from_config(config)
            else:
                return AutoModelForSequenceClassification.from_config(config)
        else:
            if framework == 'tensorflow':
                return TFAutoModel.from_config(config)
            else:
                return AutoModel.from_config(config)

    @staticmethod
    def _load_custom_model(module_path: str, weights: str, class_name: str):
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        model = model_class()
        if weights:
            if weights.endswith('.pt'):
                model.load_state_dict(torch.load(weights))
            elif weights.endswith('.h5'):
                model.load_weights(weights)
        return model
