import importlib
import os
from pathlib import Path
from typing import Union, Dict, Any
import torch
import tensorflow as tf
from transformers import AutoModel, AutoConfig, TFAutoModel

class ModelLoader:
    """Universal model loader supporting PyTorch, TensorFlow, and HuggingFace models"""
    
    @staticmethod
    def load_model(model_config: Dict[str, Any]) -> Union[torch.nn.Module, tf.keras.Model]:
        model_type = model_config.get('type', 'pytorch')
        model_name = model_config['name']
        from_pretrained = model_config.get('from_pretrained', True)
        custom_weights = model_config.get('weights')
        model_class = model_config.get('class_name')
        
        if model_type == 'pytorch':
            return ModelLoader._load_pytorch_model(model_name, from_pretrained, custom_weights, model_class)
        elif model_type == 'tensorflow':
            return ModelLoader._load_tf_model(model_name, from_pretrained, custom_weights, model_class)
        elif model_type == 'huggingface':
            return ModelLoader._load_hf_model(model_name, from_pretrained, model_config.get('framework', 'pytorch'))
        elif model_type == 'custom':
            return ModelLoader._load_custom_model(model_name, custom_weights, model_class)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def _load_pytorch_model(name: str, from_pretrained: bool, weights: str, class_name: str = None):
        if class_name:
            return ModelLoader._load_custom_model(name, weights, class_name)
            
        if from_pretrained:
            model = AutoModel.from_pretrained(name)
        else:
            config = AutoConfig.from_pretrained(name)
            model = AutoModel.from_config(config)
        
        if weights:
            state_dict = torch.load(weights)
            model.load_state_dict(state_dict)
        return model

    @staticmethod
    def _load_tf_model(name: str, from_pretrained: bool, weights: str, class_name: str = None):
        if class_name:
            return ModelLoader._load_custom_model(name, weights, class_name)
            
        if from_pretrained:
            model = TFAutoModel.from_pretrained(name)
        else:
            config = AutoConfig.from_pretrained(name)
            model = TFAutoModel.from_config(config)
        
        if weights:
            model.load_weights(weights)
        return model

    @staticmethod
    def _load_hf_model(name: str, from_pretrained: bool, framework: str):
        if from_pretrained:
            return AutoModel.from_pretrained(name, from_tf=framework == 'tf')
        config = AutoConfig.from_pretrained(name)
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
