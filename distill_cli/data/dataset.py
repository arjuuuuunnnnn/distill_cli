import os
from pathlib import Path
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import tensorflow as tf

SUPPORTED_FORMATS = ['csv', 'json', 'parquet', 'txt']

class DistillationDataset:
    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get('tokenizer_name', config['teacher']['name']))
        
    def load_data(self):
        data_config = self.config['data']
        if data_config['type'] == 'huggingface':
            return self._load_hf_dataset(data_config)
        elif data_config['type'] == 'custom':
            return self._load_custom_dataset(data_config)
        else:
            raise ValueError(f"Unsupported dataset type: {data_config['type']}")
    
    def _load_hf_dataset(self, config):
        dataset = load_dataset(config['name'], config.get('config'))
        return self._process_data(dataset)
    
    def _load_custom_dataset(self, config):
        path = Path(config['path'])
        if not path.exists():
            raise FileNotFoundError(f"Dataset path {path} does not exist")
            
        if path.is_dir():
            data_files = {
                'train': str(path / 'train.*'),
                'validation': str(path / 'val.*')
            }
            dataset = load_dataset(str(path), data_files=data_files)
        else:
            ext = path.suffix[1:]
            if ext not in SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file format: {ext}. Supported: {SUPPORTED_FORMATS}")
            
            df = pd.read_csv(path) if ext == 'csv' else \
                 pd.read_json(path) if ext == 'json' else \
                 pd.read_parquet(path) if ext == 'parquet' else \
                 pd.read_csv(path, sep='\t')  # For txt files
                 
            dataset = Dataset.from_pandas(df)
            
        return self._process_data(dataset)
    
    def _process_data(self, dataset):
        text_column = self.config['data'].get('text_column', 'text')
        label_column = self.config['data'].get('label_column', 'label')

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        def tokenize_fn(examples):
            return self.tokenizer(
                examples[text_column],
                padding='max_length',
                truncation=True,
                max_length=self.config.get('max_length', 128),
                return_tensors='np' if self.config.get('framework') == 'tensorflow' else 'pt'
            )
        
        dataset = dataset.map(tokenize_fn, batched=True)
        if label_column in dataset.column_names:
            dataset = dataset.rename_column(label_column, 'labels')
        else:
            import numpy as np
            def add_dummy_labels(examples):
                examples['labels'] = np.zeros(len(examples['input_ids']))
                return examples
            dataset = dataset.map(add_dummy_labels, batched=True)
        
        if self.config.get('framework') == 'tensorflow':
            dataset.set_format(type='tensorflow', columns=['input_ids', 'attention_mask', 'labels'])
        else:
            dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            
        return dataset
    
    def get_dataloaders(self, dataset):
        framework = self.config.get('framework', 'pytorch')
        batch_size = self.config['training']['batch_size']
        
        if framework == 'pytorch':
            return self._get_torch_dataloaders(dataset, batch_size)
        else:
            return self._get_tf_dataloaders(dataset, batch_size)
    
    def _get_torch_dataloaders(self, dataset, batch_size):
        # Check if we have a DatasetDict with predefined splits
        if isinstance(dataset, dict) or hasattr(dataset, 'keys'):
        # DatasetDict case
            if 'train' in dataset and 'test' in dataset:
                train_dataset = dataset['train']
                val_dataset = dataset['test']
            elif 'train' in dataset and 'validation' in dataset:
                train_dataset = dataset['train']
                val_dataset = dataset['validation']
            else:
                # Only have train data, create a validation split
                train_dataset = dataset['train']
                train_val_split = train_dataset.train_test_split(test_size=0.1)
                train_dataset = train_val_split['train']
                val_dataset = train_val_split['test']
        else:
            # Single Dataset case
            split = dataset.train_test_split(test_size=0.1)
            train_dataset = split['train']
            val_dataset = split['test']
    
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )
        return train_loader, val_loader
    
    def _get_tf_dataloaders(self, dataset, batch_size):
        def gen():
            for ex in dataset:
                yield ({'input_ids': ex['input_ids'], 
                       'attention_mask': ex['attention_mask']}, 
                       ex['labels'])
        
        signature = (
            {'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
             'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32)},
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
        
        tf_dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=signature
        ).batch(batch_size)
        
        train_size = int(0.9 * len(tf_dataset))
        train_ds = tf_dataset.take(train_size)
        val_ds = tf_dataset.skip(train_size)
        
        return train_ds, val_ds
