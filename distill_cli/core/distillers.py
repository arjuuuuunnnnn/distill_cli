from abc import ABC, abstractmethod
import torch
import tensorflow as tf
from typing import Dict, Any
from tqdm import tqdm

class BaseDistiller(ABC):
    """Abstract base class for model distillers"""
    
    def __init__(self, teacher, student, config: Dict[str, Any]):
        self.teacher = teacher
        self.student = student
        self.config = config
        self._freeze_teacher()
        
    def _freeze_teacher(self):
        if isinstance(self.teacher, torch.nn.Module):
            for param in self.teacher.parameters():
                param.requires_grad = False
        elif isinstance(self.teacher, tf.keras.Model):
            self.teacher.trainable = False
            
    @abstractmethod
    def compute_loss(self, student_outputs, teacher_outputs, labels):
        pass
    
    @abstractmethod
    def train_step(self, batch):
        pass
    
    @abstractmethod
    def distill(self, train_loader, val_loader):
        pass
    
    def get_logits(self, outputs):
        """Helper method to get logits from model outputs"""
        # For Hugging Face classification models
        if hasattr(outputs, 'logits'):
            return outputs.logits
        # For base Hugging Face models (add a classification head if needed)
        elif hasattr(outputs, 'last_hidden_state'):
            # This is a simplified version - you might need to implement 
            # a proper classification head for your specific case
            return outputs.last_hidden_state[:, 0, :]
        # For standard PyTorch/TF models that directly return logits
        else:
            return outputs

class PyTorchDistiller(BaseDistiller):
    def __init__(self, teacher, student, config):
        super().__init__(teacher, student, config)
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.get('learning_rate', 1e-4)
        )
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.student.to(self.device)
        self.teacher.to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def compute_loss(self, student_outputs, teacher_outputs, labels):
        student_logits = self.get_logits(student_outputs)
        teacher_logits = self.get_logits(teacher_outputs)
        
        # If the model outputs hidden states rather than logits directly,
        # we need to handle those differently
        if student_logits.shape[-1] != labels.max() + 1:
            # This is a simplified classification head - you may need a more complex one
            if not hasattr(self, 'classifier_head'):
                self.classifier_head = torch.nn.Linear(student_logits.shape[-1], labels.max() + 1).to(self.device)
                self.teacher_classifier_head = torch.nn.Linear(teacher_logits.shape[-1], labels.max() + 1).to(self.device)
                # Freeze teacher classifier head
                for param in self.teacher_classifier_head.parameters():
                    param.requires_grad = False
            
            student_logits = self.classifier_head(student_logits)
            with torch.no_grad():
                teacher_logits = self.teacher_classifier_head(teacher_logits)
        
        ce_loss = self.loss_fn(student_logits, labels)
        kl_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits / self.config['temperature'], dim=-1),
            torch.nn.functional.softmax(teacher_logits / self.config['temperature'], dim=-1),
            reduction='batchmean') * self.config['temperature']**2
        return ce_loss * self.config['alpha'] + kl_loss * (1 - self.config['alpha'])

    def train_step(self, batch):
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(self.device)
        
        self.optimizer.zero_grad()
        
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            
        student_outputs = self.student(**inputs)
        
        loss = self.compute_loss(student_outputs, teacher_outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def distill(self, train_loader, val_loader):
        for epoch in range(self.config['epochs']):
            self.student.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for batch in progress_bar:
                loss = self.train_step(batch)
                total_loss += loss
                progress_bar.set_postfix({'loss': f"{loss:.4f}"})
            
            avg_loss = total_loss / len(train_loader)
            val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")

    def evaluate(self, val_loader):
        self.student.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                teacher_outputs = self.teacher(**inputs)
                student_outputs = self.student(**inputs)
                loss = self.compute_loss(student_outputs, teacher_outputs, labels)
                total_loss += loss.item()
        return total_loss / len(val_loader)

class TensorFlowDistiller(BaseDistiller):
    def __init__(self, teacher, student, config):
        super().__init__(teacher, student, config)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.get('learning_rate', 1e-4))
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
        
        # Create classifier heads if needed
        self.classifier_head = None
        self.teacher_classifier_head = None

    def get_logits_tf(self, outputs):
        """Helper method to get logits from TF model outputs"""
        # For Hugging Face TF models
        if hasattr(outputs, 'logits'):
            return outputs.logits
        # For base Hugging Face TF models
        elif hasattr(outputs, 'last_hidden_state'):
            if self.classifier_head is None:
                hidden_size = outputs.last_hidden_state.shape[-1]
                num_labels = self.config.get('num_labels', 2)  # Default to binary classification
                self.classifier_head = tf.keras.layers.Dense(num_labels)
                self.teacher_classifier_head = tf.keras.layers.Dense(num_labels)
                # Freeze teacher classifier head
                self.teacher_classifier_head.trainable = False
            
            # Use the first token representation ([CLS])
            return self.classifier_head(outputs.last_hidden_state[:, 0, :])
        # For standard TF models
        else:
            return outputs

    def compute_loss(self, student_outputs, teacher_outputs, labels):
        student_logits = self.get_logits_tf(student_outputs)
        teacher_logits = self.get_logits_tf(teacher_outputs)
        
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, student_logits, from_logits=True)
        
        # Apply KL divergence for distillation
        student_probs = tf.nn.softmax(student_logits / self.config['temperature'])
        teacher_probs = tf.nn.softmax(teacher_logits / self.config['temperature'])
        
        kl_loss = tf.keras.losses.kullback_leibler_divergence(
            teacher_probs, student_probs) * self.config['temperature']**2
            
        return ce_loss * self.config['alpha'] + kl_loss * (1 - self.config['alpha'])

    @tf.function
    def train_step(self, batch):
        inputs = {k: v for k, v in batch.items() if k != 'labels'}
        labels = batch['labels']
        
        with tf.GradientTape() as tape:
            teacher_outputs = self.teacher(**inputs, training=False)
            student_outputs = self.student(**inputs, training=True)
            loss = self.compute_loss(student_outputs, teacher_outputs, labels)
            
        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        self.train_loss_metric.update_state(loss)
        return loss

    def distill(self, train_loader, val_loader):
        for epoch in range(self.config['epochs']):
            self.train_loss_metric.reset_states()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            
            for batch in progress_bar:
                loss = self.train_step(batch)
                progress_bar.set_postfix({'loss': f"{loss.numpy():.4f}"})
            
            val_loss = self.evaluate(val_loader)
            print(f"Epoch {epoch+1} - Train Loss: {self.train_loss_metric.result():.4f} - Val Loss: {val_loss:.4f}")

    def evaluate(self, val_loader):
        self.val_loss_metric.reset_states()
        for batch in val_loader:
            inputs = {k: v for k, v in batch.items() if k != 'labels'}
            labels = batch['labels']
            teacher_outputs = self.teacher(**inputs, training=False)
            student_outputs = self.student(**inputs, training=False)
            loss = self.compute_loss(student_outputs, teacher_outputs, labels)
            self.val_loss_metric.update_state(loss)
        return self.val_loss_metric.result().numpy()
