# Model Distillation CLI

A powerful command-line interface for knowledge distillation of machine learning models, supporting both PyTorch and TensorFlow frameworks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [JSON/YAML Configuration](#jsonyaml-configuration)
  - [Sample Configuration](#sample-configuration)
- [Usage](#usage)
  - [Distillation](#distillation)
  - [Evaluation](#evaluation)
- [Supported Models](#supported-models)
- [Data Loading](#data-loading)
- [Distillation Process](#distillation-process)
- [CLI Commands Reference](#cli-commands-reference)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

Distillation CLI provides an easy-to-use interface for model distillation, allowing you to compress larger "teacher" models into smaller "student" models while maintaining performance. The tool supports both PyTorch and TensorFlow frameworks and is compatible with Hugging Face's Transformers library.

## Features

- **Framework Agnostic**: Supports both PyTorch and TensorFlow models
- **Hugging Face Integration**: Seamless support for Hugging Face Transformers models
- **Flexible Configuration**: Configure distillation via YAML/JSON files or command-line arguments
- **Custom Models**: Support for loading custom model architectures
- **Early Stopping**: Prevents overfitting with patience-based early stopping
- **Comprehensive Evaluation**: Tools to evaluate distilled models
- **Multi-format Data Loading**: Support for various data formats and sources

## Installation

Install from source:

```bash
git clone https://github.com/arjuuuuunnnnn/distill_cli.git
cd distill_cli
pip install -e .
```


## Usage

Basic Commands:

The CLI offers three main commands:

`distill`: Run distillation using a YAML/JSON configuration file
`distill-cli`: Run distillation using command-line arguments
`evaluate`: Evaluate a distilled model


## Quick Start

Distill a BERT model into a smaller version using a configuration file:

```bash
distill-cli distill --config config.json
```

## Configuration

### JSON/YAML Configuration

The distillation process can be configured using either JSON or YAML files. Here's an example structure:

```yaml
teacher:
  name: model_name_or_path
  type: huggingface|pytorch|tensorflow
  from_pretrained: true|false
  task: sequence-classification

student:
  name: model_name_or_path
  type: huggingface|pytorch|tensorflow
  from_pretrained: true|false
  config_overrides:
    hidden_size: 512
    intermediate_size: 1024
    num_hidden_layers: 6
    num_attention_heads: 8

data:
  type: huggingface|custom
  name: dataset_name
  config: dataset_config
  text_column: text
  label_column: label

framework: pytorch|tensorflow

training:
  epochs: 10
  batch_size: 16
  learning_rate: 1e-4
  temperature: 2.0
  alpha: 0.5
  optimizer: adamw
  patience: 3

output_dir: ./distilled_model
```

### Sample Configuration

Here's a complete example for distilling BERT-base into a smaller version:

```json
{
  "teacher": {
    "name": "bert-base-uncased",
    "type": "huggingface",
    "from_pretrained": true,
    "task": "sequence-classification"
  },
  "student": {
    "name": "bert-base-uncased",
    "type": "huggingface",
    "task": "sequence-classification",
    "from_pretrained": false,
    "config_overrides": {
      "hidden_size": 512,
      "intermediate_size": 1024,
      "num_hidden_layers": 6,
      "num_attention_heads": 8
    }
  },
  "data": {
    "type": "huggingface",
    "name": "wikitext",
    "config": "wikitext-2-raw-v1",
    "text_column": "text"
  },
  "framework": "pytorch",
  "training": {
    "epochs": 1,
    "batch_size": 16,
    "learning_rate": 3e-5,
    "temperature": 2.0,
    "alpha": 0.5,
    "optimizer": "adamw",
    "patience": 3
  },
  "output_dir": "./distilled_model"
}
```

## Usage

### Distillation

Run distillation using a configuration file:

```bash
distill-cli distill --config path/to/config.json
```

Or with command-line arguments:

```bash
distill-cli distill-cli \
  --teacher bert-base-uncased \
  --student bert-base-uncased \
  --train-data ./data/train.csv \
  --val-data ./data/val.csv \
  --epochs 3 \
  --batch-size 16 \
  --lr 3e-5 \
  --temperature 2.0 \
  --alpha 0.5 \
  --output-dir ./distilled_model
```

### Evaluation

Evaluate a distilled model:

```bash
distill-cli evaluate \
  --model ./distilled_model/student_model.pt \
  --data ./data/test.csv \
  --batch-size 32 \
  --output ./evaluation_results.json
```

## Supported Models

The tool supports the following model types:

- **Hugging Face Transformers**: BERT, RoBERTa, DistilBERT, GPT-2, etc.
- **PyTorch Models**: Custom PyTorch models
- **TensorFlow Models**: Custom TensorFlow models

## Data Loading

The CLI supports multiple data sources:

- **Hugging Face Datasets**: Load directly from the Hugging Face Hub
- **Local Files**: CSV, JSON, Parquet, TXT formats
- **Custom Datasets**: Support for custom dataset implementations

## Distillation Process

The distillation process follows these steps:

1. **Model Loading**: Load teacher and student models
2. **Data Preparation**: Prepare datasets for training and evaluation
3. **Training Loop**: 
   - Teacher model produces soft targets
   - Student model is trained to match teacher outputs
   - Combined loss function (CE + KL divergence) guides learning
4. **Evaluation**: Periodic evaluation on validation data
5. **Early Stopping**: Stop training when validation loss plateaus
6. **Model Saving**: Save the distilled model to disk

## CLI Commands Reference

### `distill`

Runs distillation using a configuration file:

```bash
distill-cli distill --config path/to/config.json
```

Options:
- `--config, -c`: Path to configuration file (required)
- `--output-dir, -o`: Output directory for saving models (default: ./distilled_model)

### `distill_cli`

Runs distillation using command-line arguments:

```bash
distill-cli distill-cli --teacher path/to/teacher --student path/to/student [OPTIONS]
```

Options:
- `--teacher, -t`: Path to teacher model (required)
- `--student, -s`: Path to student model (required)
- `--train-data`: Path to training data (required)
- `--val-data`: Path to validation data (required)
- `--config, -c`: Optional path to base configuration file
- `--epochs, -e`: Number of training epochs (default: 10)
- `--batch-size, -b`: Batch size (default: 32)
- `--lr, --learning-rate`: Learning rate (default: 0.001)
- `--temperature`: Distillation temperature (default: 2.0)
- `--alpha`: Weight for CE loss vs KL loss (default: 0.5)
- `--compression`: Compression ratio (default: 0.5)
- `--output-dir, -o`: Output directory (default: ./distilled)

### `evaluate`

Evaluates a distilled model:

```bash
distill-cli evaluate --model path/to/model --data path/to/data [OPTIONS]
```

Options:
- `--model, -m`: Path to model file (required)
- `--data, -d`: Path to evaluation data (required)
- `--batch-size, -b`: Batch size (default: 32)
- `--output, -o`: Output file for results (default: ./eval_results.json)

## Advanced Usage

### Custom Model Distillation

For custom model architectures, specify the module path and class name:

```json
{
  "teacher": {
    "name": "my_module.teacher_model",
    "type": "custom",
    "class_name": "TeacherModel",
    "weights": "path/to/weights.pt"
  },
  "student": {
    "name": "my_module.student_model",
    "type": "custom",
    "class_name": "StudentModel",
    "weights": null
  }
}
```

### Mixed Framework Distillation

You can distill across frameworks, e.g., TensorFlow teacher to PyTorch student:

```json
{
  "teacher": {
    "name": "tf_bert_model",
    "type": "tensorflow",
    "from_pretrained": true
  },
  "student": {
    "name": "bert-base-uncased",
    "type": "pytorch",
    "from_pretrained": false
  },
  "framework": "pytorch"
}
```

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory error  
**Solution**: Reduce batch size or use gradient accumulation

**Issue**: Model incompatibility in distillation  
**Solution**: Ensure teacher and student models have compatible output spaces

**Issue**: Slow convergence  
**Solution**: Adjust learning rate, temperature, or alpha values

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
