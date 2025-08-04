import os
from dataclasses import dataclass
from typing import List, Optional
import yaml
import json

@dataclass
class TrainingConfig:
    """Configuration for training disaster detection models"""
    
    dataset_path: str = "data/aider_ dataset/"
    classes: List[str] = None
    train_ratio: float = 0.8
    input_size: int = 224
    
    model_name: str = 'mobilenet_v2'  
    num_classes: int = 2
    pretrained: bool = True
    freeze_backbone: bool = True
    
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    step_size: int = 20
    gamma: float = 0.1
    
    augment: bool = True
    
    num_workers: int = 4
    pin_memory: bool = True
    use_class_weights: bool = True
    
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    early_stopping_patience: int = 10
    
    log_interval: int = 10
    val_interval: int = 1
    
    export_onnx: bool = True
    export_torchscript: bool = True
    
    def __post_init__(self):
        """Set default classes if not provided"""
        if self.classes is None:
            self.classes = ['fire', 'normal']
        
        self.num_classes = len(self.classes)
        
        os.makedirs(self.save_dir, exist_ok=True)
    
    def save_config(self, path: str):
        """Save configuration to file"""
        config_dict = {
            'dataset_path': self.dataset_path,
            'classes': self.classes,
            'train_ratio': self.train_ratio,
            'input_size': self.input_size,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'freeze_backbone': self.freeze_backbone,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'step_size': self.step_size,
            'gamma': self.gamma,
            'augment': self.augment,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'use_class_weights': self.use_class_weights,
            'save_dir': self.save_dir,
            'save_best_only': self.save_best_only,
            'early_stopping_patience': self.early_stopping_patience,
            'log_interval': self.log_interval,
            'val_interval': self.val_interval,
            'export_onnx': self.export_onnx,
            'export_torchscript': self.export_torchscript
        }
        
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, path: str):
        """Load configuration from file"""
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(path, 'r') as f:
                config_dict = json.load(f)
        
        return cls(**config_dict)

def get_fire_detection_config():
    """Get configuration for fire detection (binary classification)"""
    return TrainingConfig(
        classes=['fire', 'normal'],
        model_name='mobilenet_v2',
        batch_size=32,
        num_epochs=30,
        learning_rate=0.001,
        freeze_backbone=True,
        augment=True
    )

def get_multi_disaster_config():
    """Get configuration for multi-class disaster detection"""
    return TrainingConfig(
        classes=['fire', 'collapsed_building', 'flooded_areas', 'normal'],
        model_name='efficientnet_b0',
        batch_size=16,
        num_epochs=50,
        learning_rate=0.0005,
        freeze_backbone=False,
        augment=True
    )

def get_lightweight_config():
    """Get configuration for lightweight model (for UAV deployment)"""
    return TrainingConfig(
        classes=['fire', 'normal'],
        model_name='lightweight_cnn',
        batch_size=64,
        num_epochs=100,
        learning_rate=0.01,
        freeze_backbone=False,
        augment=True,
        weight_decay=1e-3
    )

def get_all_classes_config():
    """Get configuration for all disaster classes"""
    return TrainingConfig(
        classes=['fire', 'collapsed_building', 'flooded_areas', 'traffic_incident', 'normal'],
        model_name='efficientnet_b0',
        batch_size=16,
        num_epochs=75,
        learning_rate=0.0003,
        freeze_backbone=False,
        augment=True,
        step_size=25,
        early_stopping_patience=15
    )

def create_experiment_configs():
    """Create different experiment configurations"""
    configs = {
        'fire_detection': get_fire_detection_config(),
        'multi_disaster': get_multi_disaster_config(),
        'lightweight': get_lightweight_config(),
        'all_classes': get_all_classes_config()
    }
    
    config_dir = "configs"
    os.makedirs(config_dir, exist_ok=True)
    
    for name, config in configs.items():
        config_path = os.path.join(config_dir, f"{name}.yaml")
        config.save_config(config_path)
        print(f"Saved {name} configuration to {config_path}")
    
    return configs

def print_config_comparison():
    """Print comparison of different configurations"""
    configs = {
        'Fire Detection': get_fire_detection_config(),
        'Multi-Disaster': get_multi_disaster_config(),
        'Lightweight': get_lightweight_config(),
        'All Classes': get_all_classes_config()
    }
    
    print("\n=== Configuration Comparison ===")
    print(f"{'Config':<15} {'Model':<15} {'Classes':<3} {'Batch':<6} {'Epochs':<7} {'LR':<8} {'Frozen'}")
    print("-" * 70)
    
    for name, config in configs.items():
        frozen = "Yes" if config.freeze_backbone else "No"
        print(f"{name:<15} {config.model_name:<15} {config.num_classes:<3} {config.batch_size:<6} {config.num_epochs:<7} {config.learning_rate:<8.4f} {frozen}")
    
    print("\nRecommendations:")
    print("- Fire Detection: Start here for binary classification")
    print("- Lightweight: Best for UAV deployment (smallest model)")
    print("- Multi-Disaster: Good balance of classes and performance")
    print("- All Classes: Most comprehensive but requires more data/training")

if __name__ == "__main__":
    create_experiment_configs()
    
    print_config_comparison()
    
    print("\n=== Testing Configuration Loading ===")
    config = TrainingConfig.load_config("configs/fire_detection.yaml")
    print(f"Loaded config: {config.model_name} with {config.num_classes} classes") 