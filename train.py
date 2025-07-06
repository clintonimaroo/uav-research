import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse

from config import TrainingConfig, get_fire_detection_config
from disaster_dataset import create_data_loaders
from models import create_model, freeze_backbone
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint

class DisasterTrainer:
    """Main trainer class for disaster detection models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.class_to_idx = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        # Logging
        self.writer = None
        self.setup_logging()
        
        print(f"Training on device: {self.device}")
        print(f"Configuration: {config.model_name} with {config.num_classes} classes")
    
    def setup_logging(self):
        """Setup tensorboard logging"""
        log_dir = os.path.join(self.config.save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
    
    def setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")
        self.train_loader, self.val_loader, self.class_to_idx = create_data_loaders(
            dataset_path=self.config.dataset_path,
            classes=self.config.classes,
            batch_size=self.config.batch_size,
            input_size=self.config.input_size,
            num_workers=self.config.num_workers,
            augment=self.config.augment,
            train_ratio=self.config.train_ratio
        )
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Class mapping: {self.class_to_idx}")
    
    def setup_model(self):
        """Setup model, optimizer, and scheduler"""
        print("Setting up model...")
        
        # Create model
        self.model = create_model(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained
        )
        
        # Freeze backbone if specified
        if self.config.freeze_backbone:
            freeze_backbone(self.model, self.config.model_name)
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.config.step_size,
            gamma=self.config.gamma
        )
        
        # Setup loss function
        if self.config.use_class_weights:
            # Calculate class weights from training data
            train_dataset = self.train_loader.dataset
            class_weights = train_dataset.get_class_weights()
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using class weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Print model info
        if hasattr(self.model, 'get_model_info'):
            model_info = self.model.get_model_info()
            print(f"Model: {model_info['model_name']}")
            print(f"Parameters: {model_info['total_params']:,}")
            print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Measure accuracy and record loss
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{top1.avg:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to tensorboard
            if batch_idx % self.config.log_interval == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), step)
                self.writer.add_scalar('Train/Accuracy', acc1.item(), step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], step)
        
        return losses.avg, top1.avg
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        losses = AverageMeter()
        top1 = AverageMeter()
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Measure accuracy and record loss
                acc1 = accuracy(outputs, targets, topk=(1,))[0]
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
        
        return losses.avg, top1.avg
    
    def save_model(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'class_to_idx': self.class_to_idx
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.save_dir, 'checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with accuracy: {self.best_val_acc:.2f}%")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        # Setup everything
        self.setup_data()
        self.setup_model()
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            if epoch % self.config.val_interval == 0:
                val_loss, val_acc = self.validate()
                
                # Log to tensorboard
                self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
                self.writer.add_scalar('Epoch/Train_Acc', train_acc, epoch)
                self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
                self.writer.add_scalar('Epoch/Val_Acc', val_acc, epoch)
                
                # Print results
                print(f'Epoch {epoch+1}/{self.config.num_epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
                # Check if this is the best model
                is_best = val_acc > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_acc
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                # Save checkpoint
                if not self.config.save_best_only or is_best:
                    self.save_model(is_best)
                
                # Early stopping
                if self.epochs_without_improvement >= self.config.early_stopping_patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break
            
            # Update learning rate
            self.scheduler.step()
        
        print(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")
        self.writer.close()
        
        return self.best_val_acc

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train disaster detection model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--model', type=str, default='mobilenet_v2', 
                       choices=['mobilenet_v2', 'efficientnet_b0', 'lightweight_cnn'],
                       help='Model architecture')
    parser.add_argument('--classes', nargs='+', default=['fire', 'normal'],
                       help='Classes to train on')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = TrainingConfig.load_config(args.config)
    else:
        config = get_fire_detection_config()
        
        # Override with command line arguments
        config.model_name = args.model
        config.classes = args.classes
        config.num_classes = len(args.classes)
        config.num_epochs = args.epochs
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        config.save_dir = args.save_dir
    
    # Save configuration
    config.save_config(os.path.join(config.save_dir, 'config.yaml'))
    
    # Create trainer and train
    trainer = DisasterTrainer(config)
    best_acc = trainer.train()
    
    print(f"\nTraining Summary:")
    print(f"Model: {config.model_name}")
    print(f"Classes: {config.classes}")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {config.save_dir}")

if __name__ == "__main__":
    main() 