import torch
import numpy as np
import cv2
from PIL import Image
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import TrainingConfig
    from models import create_model
    from disaster_dataset import get_transforms
except ImportError:
    print("Warning: Disaster classifier components not available")
    TrainingConfig = None

class DisasterClassifierBridge:
    def __init__(self, model_path=None, config_path=None):
        self.model = None
        self.transform = None
        self.class_to_idx = None
        self.idx_to_class = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path and TrainingConfig:
            self._load_model(model_path, config_path)
    
    def _load_model(self, model_path, config_path=None):
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if config_path:
                config = TrainingConfig.load_config(config_path)
            else:
                config = checkpoint.get('config')
            
            self.class_to_idx = checkpoint['class_to_idx']
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            
            self.model = create_model(
                model_name=config.model_name,
                num_classes=config.num_classes
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            _, self.transform = get_transforms(config.input_size, augment=False)
            
            print(f"Loaded disaster classifier: {config.model_name}")
            
        except Exception as e:
            print(f"Failed to load disaster classifier: {e}")
            self.model = None
    
    def predict_disaster(self, image_region):
        if self.model is None:
            return False
        
        try:
            if isinstance(image_region, np.ndarray):
                if image_region.shape[-1] == 3:
                    if image_region.max() <= 1.0:
                        image_region = (image_region * 255).astype(np.uint8)
                    image = Image.fromarray(image_region)
                else:
                    image = Image.fromarray((image_region * 255).astype(np.uint8))
                    image = image.convert('RGB')
            else:
                image = image_region
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = self.idx_to_class[predicted_idx.item()]
                confidence_score = confidence.item()
                
                is_disaster = predicted_class != 'normal' and confidence_score > 0.7
                
                return is_disaster
                
        except Exception as e:
            print(f"Error in disaster prediction: {e}")
            return False
    
    def get_disaster_type(self, image_region):
        if self.model is None:
            return 'unknown', 0.0
        
        try:
            if isinstance(image_region, np.ndarray):
                if image_region.shape[-1] == 3:
                    if image_region.max() <= 1.0:
                        image_region = (image_region * 255).astype(np.uint8)
                    image = Image.fromarray(image_region)
                else:
                    image = Image.fromarray((image_region * 255).astype(np.uint8))
                    image = image.convert('RGB')
            else:
                image = image_region
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_class = self.idx_to_class[predicted_idx.item()]
                confidence_score = confidence.item()
                
                return predicted_class, confidence_score
                
        except Exception as e:
            print(f"Error in disaster type prediction: {e}")
            return 'unknown', 0.0

class SimulatedDisasterDetector:
    def __init__(self):
        self.disaster_types = ['fire', 'collapsed_building', 'flooded_areas', 'traffic_incident']
        
    def predict_disaster(self, image_region):
        hazard_detected = np.random.random() < 0.1
        return hazard_detected
    
    def get_disaster_type(self, image_region):
        if self.predict_disaster(image_region):
            disaster_type = np.random.choice(self.disaster_types)
            confidence = 0.8 + np.random.random() * 0.2
            return disaster_type, confidence
        return 'normal', 0.9

def create_disaster_detector(model_path=None, config_path=None, use_real_model=True):
    if use_real_model and model_path:
        detector = DisasterClassifierBridge(model_path, config_path)
        if detector.model is not None:
            return detector
    
    print("Using simulated disaster detector")
    return SimulatedDisasterDetector()