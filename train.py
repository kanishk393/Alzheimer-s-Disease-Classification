#!/usr/bin/env python3
"""
Training script for Alzheimer's disease classification model.

This script handles the complete training pipeline including data loading,
model training, validation, and checkpointing.

Usage:
    python scripts/train.py --config config.yaml
    python scripts/train.py --config config.yaml --resume models/checkpoints/checkpoint.pth
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from alzheimer_classifier.models.efficientnet_model import create_model
from alzheimer_classifier.utils.logging_utils import setup_logging
from alzheimer_classifier.utils.checkpoint_utils import save_checkpoint, load_checkpoint
from alzheimer_classifier.training.metrics import calculate_metrics
from alzheimer_classifier.evaluation.visualization import plot_training_curves


class AlzheimerTrainer:
    """
    Training class for Alzheimer's disease classification.
    
    This class handles the complete training pipeline including:
    - Data loading and preprocessing
    - Model training and validation
    - Checkpointing and logging
    - Metric calculation and visualization
    """
    
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.best_val_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Setup logging
        setup_logging(config['logging'])
        self.logger = logging.getLogger(__name__)
        
        # Setup experiment tracking
        self._setup_tracking()
        
        # Initialize model, dataloaders, optimizer, scheduler
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        self.logger.info("Trainer initialized successfully")
    
    def _get_device(self):
        """Determine the appropriate device for training."""
        if self.config['environment']['device'] == 'auto':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config['environment']['device'])
        
        self.logger.info(f"Using device: {device}")
        return device
    
    def _setup_tracking(self):
        """Setup experiment tracking with TensorBoard and optionally Weights & Biases."""
        # TensorBoard
        if self.config['logging']['tensorboard']['enabled']:
            log_dir = self.config['logging']['tensorboard']['log_dir']
            os.makedirs(log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir)
        else:
            self.tb_writer = None
        
        # Weights & Biases
        if self.config['logging']['wandb']['enabled']:
            wandb.init(
                project=self.config['logging']['wandb']['project'],
                entity=self.config['logging']['wandb']['entity'],
                config=self.config
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
    
    def _setup_model(self):
        """Initialize the model."""
        self.model = create_model(
            num_classes=self.config['model']['num_classes'],
            pretrained=self.config['model']['pretrained'],
            dropout_rate=self.config['model']['dropout_rate'],
            device=self.device
        )
        
        model_info = self.model.get_model_info()
        self.logger.info(f"Model info: {model_info}")
    
    def _setup_data(self):
        """Setup data loaders for training and validation."""
        # Define transforms
        train_transforms = transforms.Compose([
            transforms.Resize((self.config['model']['input_size'], 
                             self.config['model']['input_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['model']['normalization']['mean'],
                std=self.config['model']['normalization']['std']
            )
        ])
        
        val_transforms = transforms.Compose([
            transforms.Resize((self.config['model']['input_size'], 
                             self.config['model']['input_size'])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config['model']['normalization']['mean'],
                std=self.config['model']['normalization']['std']
            )
        ])
        
        # Load datasets
        full_dataset = datasets.ImageFolder(
            root=self.config['data']['augmented_path'],
            transform=train_transforms
        )
        
        # Split dataset
        train_size = int(self.config['data']['train_split'] * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config['environment']['seed'])
        )
        
        # Update validation dataset transform
        val_dataset.dataset.transform = val_transforms
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data']['pin_memory']
        )
        
        self.logger.info(f"Train dataset size: {len(train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(val_dataset)}")
        self.logger.info(f"Classes: {full_dataset.classes}")
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        optimizer_name = self.config['training']['optimizer']
        
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        self.logger.info(f"Optimizer: {optimizer_name}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_config = self.config['training']['scheduler']
        
        if scheduler_config['type'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_config['patience'],
                factor=scheduler_config['factor'],
                min_lr=scheduler_config['min_lr']
            )
        elif scheduler_config['type'] == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Scheduler: {scheduler_config['type']}")
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clipping']['enabled']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clipping']['max_norm']
                )
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100 * correct / total:.2f}%"
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_accuracy': self.best_val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved with accuracy: {self.best_val_accuracy:.2f}%")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # TensorBoard logging
            if self.tb_writer:
                self.tb_writer.add_scalar('Loss/Train', train_loss, epoch)
                self.tb_writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.tb_writer.add_scalar('Accuracy/Train', train_acc, epoch)
                self.tb_writer.add_scalar('Accuracy/Validation', val_acc, epoch)
                self.tb_writer.add_scalar('Learning_Rate', 
                                        self.optimizer.param_groups[0]['lr'], epoch)
            
            # Weights & Biases logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_acc > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_acc
            
            self.save_checkpoint(epoch, is_best)
        
        self.logger.info(f"Training completed! Best validation accuracy: {self.best_val_accuracy:.2f}%")
        
        # Plot training curves
        plot_training_curves(
            self.train_losses, self.val_losses,
            self.train_accuracies, self.val_accuracies,
            save_path="results/figures/training_curves.png"
        )
        
        # Close TensorBoard writer
        if self.tb_writer:
            self.tb_writer.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Alzheimer's disease classification model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    torch.manual_seed(config['environment']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['environment']['seed'])
    
    # Create trainer
    trainer = AlzheimerTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint['scheduler_state_dict']:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.best_val_accuracy = checkpoint['best_val_accuracy']
        trainer.train_losses = checkpoint['train_losses']
        trainer.val_losses = checkpoint['val_losses']
        trainer.train_accuracies = checkpoint['train_accuracies']
        trainer.val_accuracies = checkpoint['val_accuracies']
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()