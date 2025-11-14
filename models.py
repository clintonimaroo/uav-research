import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

try:
    from torchvision.models import MobileNet_V2_Weights, EfficientNet_B0_Weights
    TORCHVISION_NEW = True
except ImportError:
    TORCHVISION_NEW = False

class DisasterClassifier(nn.Module):
    """Base class for disaster detection models"""
    
    def __init__(self, num_classes: int = 2, model_name: str = 'mobilenet_v2'):
        super(DisasterClassifier, self).__init__()
        self.num_classes = num_classes
        self.model_name = model_name
        
        if model_name == 'mobilenet_v2':
            self.model = self._create_mobilenet_v2()
        elif model_name == 'efficientnet_b0':
            self.model = self._create_efficientnet_b0()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _create_mobilenet_v2(self):
        """Create MobileNetV2 model for disaster detection"""
        if TORCHVISION_NEW:
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            model = models.mobilenet_v2(pretrained=True)
        
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, self.num_classes)
        )
        
        return model
    
    def _create_efficientnet_b0(self):
        """Create EfficientNet-B0 model for disaster detection"""
        if TORCHVISION_NEW:
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            model = models.efficientnet_b0(pretrained=True)
        
        model.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(model.classifier[1].in_features, self.num_classes)
        )
        
        return model
    
    def forward(self, x):
        return self.model(x)
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }

class LightweightCNN(nn.Module):
    """Custom lightweight CNN for UAV deployment"""
    
    def __init__(self, num_classes: int = 2, input_channels: int = 3):
        super(LightweightCNN, self).__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'lightweight_cnn',
            'num_classes': self.num_classes,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }

def create_model(model_name: str = 'mobilenet_v2', 
                num_classes: int = 2,
                pretrained: bool = True) -> nn.Module:
    """Factory function to create disaster detection models"""
    
    if model_name in ['mobilenet_v2', 'efficientnet_b0']:
        model = DisasterClassifier(num_classes=num_classes, model_name=model_name)
    elif model_name == 'lightweight_cnn':
        model = LightweightCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model

def freeze_backbone(model: nn.Module, model_name: str):
    """Freeze backbone layers for fine-tuning"""
    
    if model_name == 'mobilenet_v2':
        # Freeze all layers except classifier
        for param in model.model.features.parameters():
            param.requires_grad = False
    elif model_name == 'efficientnet_b0':
        # Freeze all layers except classifier
        for param in model.model.features.parameters():
            param.requires_grad = False
    elif model_name == 'lightweight_cnn':
        # For custom CNN, freeze first few layers
        for param in model.features[:6].parameters():
            param.requires_grad = False
    
    print(f"Backbone frozen for {model_name}")

def unfreeze_backbone(model: nn.Module):
    """Unfreeze all layers for full fine-tuning"""
    for param in model.parameters():
        param.requires_grad = True
    print("All layers unfrozen")

def get_model_comparison():
    """Compare different model architectures"""
    models_info = []
    
    for model_name in ['mobilenet_v2', 'efficientnet_b0', 'lightweight_cnn']:
        model = create_model(model_name, num_classes=2)
        info = model.get_model_info()
        models_info.append(info)
    
    print("\n=== Model Comparison ===")
    print(f"{'Model':<20} {'Params (M)':<12} {'Size (MB)':<12} {'Suitable for UAV'}")
    print("-" * 60)
    
    for info in models_info:
        params_m = info['total_params'] / 1e6
        size_mb = info['model_size_mb']
        suitable = "✓" if size_mb < 50 else "⚠️" if size_mb < 100 else "✗"
        
        print(f"{info['model_name']:<20} {params_m:<12.2f} {size_mb:<12.2f} {suitable}")
    
    return models_info

def test_models():
    """Test model creation and forward pass"""
    print("Testing model architectures...")
    
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    for model_name in ['mobilenet_v2', 'efficientnet_b0', 'lightweight_cnn']:
        print(f"\nTesting {model_name}...")
        
        model = create_model(model_name, num_classes=2)
        model.eval()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model info: {model.get_model_info()}")
        
        freeze_backbone(model, model_name)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params after freezing: {trainable_params:,}")
        
        unfreeze_backbone(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params after unfreezing: {trainable_params:,}")

if __name__ == "__main__":
    test_models()
    get_model_comparison() 