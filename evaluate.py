import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import argparse

from config import TrainingConfig
from disaster_dataset import create_data_loaders, AIDERDataset, get_transforms
from models import create_model
from utils import load_checkpoint, calculate_metrics, create_confusion_matrix

class ModelEvaluator:
    """Comprehensive model evaluation for disaster detection"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load configuration
        if config_path:
            self.config = TrainingConfig.load_config(config_path)
        else:
            self.config = self.checkpoint.get('config', None)
            if self.config is None:
                raise ValueError("No configuration found. Please provide config_path.")
        
        self.class_to_idx = self.checkpoint.get('class_to_idx', {})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Initialize model
        self.model = create_model(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes
        )
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model: {self.config.model_name}")
        print(f"Classes: {list(self.class_to_idx.keys())}")
        print(f"Best validation accuracy: {self.checkpoint.get('best_val_acc', 'N/A'):.2f}%")
    
    def evaluate_dataset(self, dataset_loader, dataset_name="Test"):
        """Evaluate model on a dataset"""
        print(f"\nEvaluating on {dataset_name} set...")
        
        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for images, targets in tqdm(dataset_loader, desc=f"Evaluating {dataset_name}"):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(dataset_loader)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        return all_preds, all_targets, all_probs, avg_loss
    
    def calculate_detailed_metrics(self, y_true, y_pred, y_probs):
        """Calculate detailed classification metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            roc_auc_score, average_precision_score
        )
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # Per-class metrics
        class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        metrics = {}
        
        for i, class_name in enumerate(class_names):
            metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
            
            # ROC AUC for binary classification or multiclass
            if len(class_names) == 2:
                if i == 1:  # Positive class
                    try:
                        metrics[class_name]['roc_auc'] = roc_auc_score(y_true, y_probs[:, i])
                        metrics[class_name]['avg_precision'] = average_precision_score(y_true, y_probs[:, i])
                    except:
                        metrics[class_name]['roc_auc'] = 0.0
                        metrics[class_name]['avg_precision'] = 0.0
            else:
                # Multiclass - one-vs-rest AUC
                try:
                    y_true_binary = (y_true == i).astype(int)
                    metrics[class_name]['roc_auc'] = roc_auc_score(y_true_binary, y_probs[:, i])
                    metrics[class_name]['avg_precision'] = average_precision_score(y_true_binary, y_probs[:, i])
                except:
                    metrics[class_name]['roc_auc'] = 0.0
                    metrics[class_name]['avg_precision'] = 0.0
        
        # Overall metrics
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        metrics['overall'] = {
            'accuracy': accuracy,
            'precision': precision_avg,
            'recall': recall_avg,
            'f1': f1_avg
        }
        
        return metrics, class_names
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf(), cm
    
    def plot_class_distribution(self, y_true, class_names, save_path=None):
        """Plot class distribution in the dataset"""
        unique, counts = np.unique(y_true, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar([class_names[i] for i in unique], counts)
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def analyze_misclassifications(self, y_true, y_pred, y_probs, class_names, top_k=5):
        """Analyze worst misclassifications"""
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return
        
        # Calculate confidence for each prediction
        confidences = np.max(y_probs, axis=1)
        
        # Get most confident misclassifications
        misclassified_confidences = confidences[misclassified_indices]
        most_confident_errors = misclassified_indices[np.argsort(misclassified_confidences)[::-1]]
        
        print(f"\nTop {min(top_k, len(most_confident_errors))} Most Confident Misclassifications:")
        print("-" * 80)
        
        for i, idx in enumerate(most_confident_errors[:top_k]):
            true_class = class_names[y_true[idx]]
            pred_class = class_names[y_pred[idx]]
            confidence = confidences[idx]
            
            print(f"{i+1}. Index {idx}: True={true_class}, Pred={pred_class}, Confidence={confidence:.3f}")
    
    def generate_report(self, save_dir=None):
        """Generate comprehensive evaluation report"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Create test dataset
        _, val_transform = get_transforms(self.config.input_size, augment=False)
        test_dataset = AIDERDataset(
            dataset_path=self.config.dataset_path,
            classes=self.config.classes,
            transform=val_transform,
            split='val',
            train_ratio=self.config.train_ratio
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=4
        )
        
        # Evaluate
        y_pred, y_true, y_probs, avg_loss = self.evaluate_dataset(test_loader, "Test")
        
        # Calculate metrics
        metrics, class_names = self.calculate_detailed_metrics(y_true, y_pred, y_probs)
        
        # Print results
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Model: {self.config.model_name}")
        print(f"Dataset: {self.config.dataset_path}")
        print(f"Test samples: {len(y_true)}")
        print(f"Average loss: {avg_loss:.4f}")
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {metrics['overall']['accuracy']:.4f}")
        print(f"  Precision: {metrics['overall']['precision']:.4f}")
        print(f"  Recall: {metrics['overall']['recall']:.4f}")
        print(f"  F1-Score: {metrics['overall']['f1']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
        print("-" * 60)
        
        for class_name in class_names:
            if class_name in metrics:
                m = metrics[class_name]
                print(f"{class_name:<15} {m['precision']:<10.3f} {m['recall']:<10.3f} "
                      f"{m['f1']:<10.3f} {m['support']:<10}")
        
        # Generate plots
        if save_dir:
            # Confusion matrix
            fig_cm, cm = self.plot_confusion_matrix(y_true, y_pred, class_names, 
                                                   os.path.join(save_dir, 'confusion_matrix.png'))
            plt.close(fig_cm)
            
            # Class distribution
            fig_dist = self.plot_class_distribution(y_true, class_names,
                                                   os.path.join(save_dir, 'class_distribution.png'))
            plt.close(fig_dist)
            
            # Save metrics to file
            import json
            metrics_file = os.path.join(save_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, dict):
                        json_metrics[k] = {k2: float(v2) if isinstance(v2, (np.float32, np.float64)) else int(v2) 
                                         for k2, v2 in v.items()}
                    else:
                        json_metrics[k] = float(v) if isinstance(v, (np.float32, np.float64)) else v
                json.dump(json_metrics, f, indent=2)
            
            print(f"\nResults saved to: {save_dir}")
        
        # Analyze misclassifications
        self.analyze_misclassifications(y_true, y_pred, y_probs, class_names)
        
        return metrics

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate disaster detection model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, args.config_path)
    
    # Generate report
    metrics = evaluator.generate_report(args.save_dir)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {args.save_dir}")

if __name__ == "__main__":
    main() 