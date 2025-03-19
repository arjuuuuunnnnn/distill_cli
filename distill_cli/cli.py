import click
import yaml
import torch
from pathlib import Path
from .core.model_loader import ModelLoader
from .core.distillers import PyTorchDistiller, TensorFlowDistiller
from .data.dataset import DistillationDataset

@click.group()
def cli():
    """Model Distillation CLI"""
    pass

@cli.command()
@click.option('--config', '-c', required=True, help='Path to config YAML file')
@click.option('--output-dir', '-o', default='./distilled_model', help='Output directory')
def distill(config, output_dir):
    """Run distillation using configuration file"""
    config = load_config(config)
    config['output_dir'] = output_dir
    
    # Load models
    teacher = ModelLoader.load_model(config['teacher'])
    student = ModelLoader.load_model(config['student'])
    
    # Prepare data
    dataset = DistillationDataset(config)
    processed_data = dataset.load_data()
    train_loader, val_loader = dataset.get_dataloaders(processed_data)
    
    # Initialize distiller
    framework = config.get('framework', 'pytorch')
    if framework == 'pytorch':
        distiller = PyTorchDistiller(teacher, student, config['training'])
    elif framework == 'tensorflow':
        distiller = TensorFlowDistiller(teacher, student, config['training'])
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    
    # Run distillation
    distiller.distill(train_loader, val_loader)
    
    # Save
