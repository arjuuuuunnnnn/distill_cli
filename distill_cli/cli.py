import click
import yaml
import json
import torch
import os
from pathlib import Path
from .core.model_loader import ModelLoader
from .core.distillers import PyTorchDistiller, TensorFlowDistiller
from .data.dataset import DistillationDataset

@click.group()
def cli():
    """Model Distillation CLI"""
    pass

def load_config(config_path):
    """Load configuration from YAML or JSON file"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    if path.suffix.lower() in ['.yaml', '.yml']:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    elif path.suffix.lower() == '.json':
        with open(path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

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
    
    # Save distilled model
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if framework == 'pytorch':
        torch.save(student.state_dict(), save_path / 'student_model.pt')
    elif framework == 'tensorflow':
        student.save_weights(str(save_path / 'student_model.h5'))
    
    print(f"Distilled model saved to {save_path}")

@cli.command()
@click.option('--teacher', '-t', required=True, help='Path to teacher model')
@click.option('--student', '-s', required=True, help='Path to student model')
@click.option('--train-data', required=True, help='Path to training data')
@click.option('--val-data', required=True, help='Path to validation data')
@click.option('--config', '-c', help='Path to config file')
@click.option('--epochs', '-e', type=int, default=10, help='Number of training epochs')
@click.option('--batch-size', '-b', type=int, default=32, help='Batch size')
@click.option('--lr', '--learning-rate', type=float, default=0.001, help='Learning rate')
@click.option('--temperature', type=float, default=2.0, help='Distillation temperature')
@click.option('--alpha', type=float, default=0.5, help='Weight for CE loss vs KL loss')
@click.option('--compression', type=float, default=0.5, help='Compression ratio')
@click.option('--output-dir', '-o', default='./distilled', help='Output directory')
def distill_cli(teacher, student, train_data, val_data, config, epochs, batch_size, 
                lr, temperature, alpha, compression, output_dir):
    """Run distillation using command line arguments"""
    # Build config from command line args
    config_dict = {
        'teacher': {
            'name': teacher,
            'type': 'pytorch' if teacher.endswith('.pt') else 'tensorflow',
            'from_pretrained': False
        },
        'student': {
            'name': student,
            'type': 'pytorch' if student.endswith('.pt') else 'tensorflow',
            'from_pretrained': False
        },
        'data': {
            'type': 'custom',
            'path': train_data,
            'validation_path': val_data
        },
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'temperature': temperature,
            'alpha': alpha,
            'compression': compression
        },
        'output_dir': output_dir
    }
    
    # If config file is provided, use it as base and override with CLI args
    if config:
        base_config = load_config(config)
        # Merge configs (CLI args take precedence)
        for key, value in base_config.items():
            if key not in config_dict:
                config_dict[key] = value
            elif isinstance(value, dict) and key in config_dict:
                config_dict[key].update(value)
    
    # Load models
    teacher_model = ModelLoader.load_model(config_dict['teacher'])
    student_model = ModelLoader.load_model(config_dict['student'])
    
    # Load and prepare data
    dataset = DistillationDataset(config_dict)
    processed_data = dataset.load_data()
    train_loader, val_loader = dataset.get_dataloaders(processed_data)
    
    # Initialize distiller
    framework = config_dict.get('framework', 'pytorch')
    if framework == 'pytorch':
        distiller = PyTorchDistiller(teacher_model, student_model, config_dict['training'])
    elif framework == 'tensorflow':
        distiller = TensorFlowDistiller(teacher_model, student_model, config_dict['training'])
    else:
        raise ValueError(f"Unsupported framework: {framework}")
    
    # Run distillation
    distiller.distill(train_loader, val_loader)
    
    # Save distilled model
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if framework == 'pytorch':
        torch.save(student_model.state_dict(), save_path / 'student_model.pt')
    elif framework == 'tensorflow':
        student_model.save_weights(str(save_path / 'student_model.h5'))
    
    # Save config
    with open(save_path / 'distillation_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Distilled model saved to {save_path}")

@cli.command()
@click.option('--model', '-m', required=True, help='Path to model file')
@click.option('--data', '-d', required=True, help='Path to evaluation data')
@click.option('--batch-size', '-b', type=int, default=32, help='Batch size')
@click.option('--output', '-o', default='./eval_results.json', help='Output file for results')
def evaluate(model, data, batch_size, output):
    """Evaluate a distilled model"""
    # Implementation for evaluation command
    model_path = Path(model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model}")
    
    model_config = {
        'name': str(model_path),
        'type': 'pytorch' if model.endswith('.pt') else 'tensorflow',
        'from_pretrained': False
    }
    
    # Load model
    loaded_model = ModelLoader.load_model(model_config)
    
    # Load data
    data_config = {
        'data': {
            'type': 'custom',
            'path': data
        },
        'framework': 'pytorch' if model.endswith('.pt') else 'tensorflow',
        'training': {
            'batch_size': batch_size
        }
    }
    
    dataset = DistillationDataset(data_config)
    processed_data = dataset.load_data()
    _, eval_loader = dataset.get_dataloaders(processed_data)
    
    # Run evaluation
    framework = data_config.get('framework', 'pytorch')
    if framework == 'pytorch':
        results = evaluate_pytorch_model(loaded_model, eval_loader)
    else:
        results = evaluate_tensorflow_model(loaded_model, eval_loader)
    
    # Save results
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {output_path}")

def evaluate_pytorch_model(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return {
        'accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct
    }

def evaluate_tensorflow_model(model, data_loader):
    model.compile(
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    results = model.evaluate(data_loader)
    
    return {
        'loss': float(results[0]),
        'accuracy': float(results[1] * 100)
    }

if __name__ == '__main__':
    cli()
