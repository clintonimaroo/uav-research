import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from config import TrainingConfig
from disaster_dataset import create_data_loaders, AIDERDataset, get_transforms
from models import create_model


def fixed_confusion_matrix(y_true, y_pred, labels):
    label_to_index = {int(label): idx for idx, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        true_index = label_to_index.get(int(true_label))
        pred_index = label_to_index.get(int(pred_label))
        if true_index is not None and pred_index is not None:
            cm[true_index, pred_index] += 1
    return cm


def accuracy_score_fixed(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def precision_recall_fscore_support_fixed(y_true, y_pred, labels, average=None):
    cm = fixed_confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    actual = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    support = actual.astype(int)

    if average is None:
        return precision, recall, f1, support
    if average == 'weighted':
        total = support.sum()
        weights = support / total if total else np.zeros_like(support, dtype=float)
        return (
            float(np.sum(precision * weights)),
            float(np.sum(recall * weights)),
            float(np.sum(f1 * weights)),
            None,
        )
    if average == 'macro':
        return float(np.mean(precision)), float(np.mean(recall)), float(np.mean(f1)), None
    raise ValueError(f"Unsupported average: {average}")


def rankdata_average(values):
    values = np.asarray(values)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def roc_auc_binary(y_true_binary, scores):
    y_true_binary = np.asarray(y_true_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos = y_true_binary == 1
    neg = y_true_binary == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC is undefined when one class is missing")
    ranks = rankdata_average(scores)
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision_binary(y_true_binary, scores):
    y_true_binary = np.asarray(y_true_binary, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((y_true_binary == 1).sum())
    if n_pos == 0:
        raise ValueError("Average precision is undefined without positives")
    order = np.argsort(-scores)
    sorted_true = y_true_binary[order]
    tp = np.cumsum(sorted_true == 1)
    precision_at_k = tp / np.arange(1, len(sorted_true) + 1)
    return float(precision_at_k[sorted_true == 1].sum() / n_pos)


def classification_report_fixed(y_true, y_pred, labels, target_names):
    precision, recall, f1, support = precision_recall_fscore_support_fixed(y_true, y_pred, labels)
    accuracy = accuracy_score_fixed(y_true, y_pred)
    macro = precision_recall_fscore_support_fixed(y_true, y_pred, labels, average='macro')
    weighted = precision_recall_fscore_support_fixed(y_true, y_pred, labels, average='weighted')

    lines = [
        f"{'':>20} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}",
        "",
    ]
    for name, p, r, f, s in zip(target_names, precision, recall, f1, support):
        lines.append(f"{name:>20} {p:>10.2f} {r:>10.2f} {f:>10.2f} {int(s):>10d}")
    lines.extend([
        "",
        f"{'accuracy':>20} {'':>10} {'':>10} {accuracy:>10.2f} {int(np.sum(support)):>10d}",
        f"{'macro avg':>20} {macro[0]:>10.2f} {macro[1]:>10.2f} {macro[2]:>10.2f} {int(np.sum(support)):>10d}",
        f"{'weighted avg':>20} {weighted[0]:>10.2f} {weighted[1]:>10.2f} {weighted[2]:>10.2f} {int(np.sum(support)):>10d}",
        "",
    ])
    return "\n".join(lines)

class ModelEvaluator:
    """Comprehensive model evaluation for disaster detection"""
    
    def __init__(self, model_path: str, config_path: str = None):
        self.model_path = model_path
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        # Load checkpoint
        self.checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if config_path:
            self.config = TrainingConfig.load_config(config_path)
        else:
            self.config = self.checkpoint.get('config', None)
            if self.config is None:
                raise ValueError("No configuration found. Please provide config_path.")
        
        self.class_to_idx = self.checkpoint.get('class_to_idx', {})
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
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
        class_names = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]
        labels = list(range(len(class_names)))
        accuracy = accuracy_score_fixed(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support_fixed(
            y_true,
            y_pred,
            labels=labels,
        )
        metrics = {}
        
        for i, class_name in enumerate(class_names):
            metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
            
            if len(class_names) == 2:
                if i == 1:  # Positive class
                    try:
                        metrics[class_name]['roc_auc'] = roc_auc_binary(y_true, y_probs[:, i])
                        metrics[class_name]['avg_precision'] = average_precision_binary(y_true, y_probs[:, i])
                    except ValueError:
                        metrics[class_name]['roc_auc'] = 0.0
                        metrics[class_name]['avg_precision'] = 0.0
            else:
                try:
                    y_true_binary = (y_true == i).astype(int)
                    metrics[class_name]['roc_auc'] = roc_auc_binary(y_true_binary, y_probs[:, i])
                    metrics[class_name]['avg_precision'] = average_precision_binary(y_true_binary, y_probs[:, i])
                except ValueError:
                    metrics[class_name]['roc_auc'] = 0.0
                    metrics[class_name]['avg_precision'] = 0.0
        
        precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support_fixed(
            y_true,
            y_pred,
            labels=labels,
            average='weighted',
        )
        
        metrics['overall'] = {
            'accuracy': accuracy,
            'precision': precision_avg,
            'recall': recall_avg,
            'f1': f1_avg
        }
        
        return metrics, class_names
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_path=None):
        """Plot confusion matrix"""
        labels = list(range(len(class_names)))
        cm = fixed_confusion_matrix(y_true, y_pred, labels=labels)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = 'white' if cm[i, j] > threshold else 'black'
                ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color=color)
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
        
        confidences = np.max(y_probs, axis=1)
        
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
        
        _, val_transform = get_transforms(self.config.input_size, augment=False)
        np.random.seed(42)
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
        
        y_pred, y_true, y_probs, avg_loss = self.evaluate_dataset(test_loader, "Test")
        
        metrics, class_names = self.calculate_detailed_metrics(y_true, y_pred, y_probs)
        
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
        
        if save_dir:
            # Confusion matrix
            fig_cm, cm = self.plot_confusion_matrix(y_true, y_pred, class_names, 
                                                   os.path.join(save_dir, 'confusion_matrix.png'))
            plt.close(fig_cm)
            
            fig_dist = self.plot_class_distribution(y_true, class_names,
                                                   os.path.join(save_dir, 'class_distribution.png'))
            plt.close(fig_dist)
            np.savetxt(
                os.path.join(save_dir, 'confusion_matrix.csv'),
                cm,
                delimiter=',',
                fmt='%d',
                header=','.join(class_names),
                comments='',
            )
            
            import json
            with open(os.path.join(save_dir, 'class_names.json'), 'w') as f:
                json.dump(class_names, f, indent=2)
            with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
                f.write(classification_report_fixed(
                    y_true,
                    y_pred,
                    labels=list(range(len(class_names))),
                    target_names=class_names,
                ))
            metrics_file = os.path.join(save_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, dict):
                        json_metrics[k] = {
                            k2: float(v2) if isinstance(v2, (np.floating, float)) else int(v2)
                            for k2, v2 in v.items()
                        }
                    else:
                        json_metrics[k] = float(v) if isinstance(v, (np.floating, float)) else v
                json.dump(json_metrics, f, indent=2)
            
            print(f"\nResults saved to: {save_dir}")
        
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
    
    evaluator = ModelEvaluator(args.model_path, args.config_path)
    
    metrics = evaluator.generate_report(args.save_dir)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {args.save_dir}")

if __name__ == "__main__":
    main() 
