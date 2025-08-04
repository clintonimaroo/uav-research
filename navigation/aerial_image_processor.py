import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TrainingConfig
from models import create_model
from disaster_dataset import get_transforms

class AerialImageProcessor:
    def __init__(self, classifier_path: str = "../checkpoints/best_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_classifier(classifier_path)
        
    def _load_classifier(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.config = checkpoint['config']
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        self.classifier = create_model(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes
        )
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
        
        _, self.transform = get_transforms(self.config.input_size, augment=False)
        
    def generate_aerial_scene(self, grid_size: int = 30) -> np.ndarray:
        aerial_image = np.zeros((grid_size * 20, grid_size * 20, 3), dtype=np.uint8)
        
        aerial_image[:, :] = [34, 139, 34]
        
        num_disasters = np.random.randint(5, 12)
        disaster_locations = []
        
        for _ in range(num_disasters):
            disaster_type = np.random.choice(['fire', 'collapsed_building', 'flooded_areas', 'traffic_incident'])
            
            center_x = np.random.randint(50, grid_size * 20 - 50)
            center_y = np.random.randint(50, grid_size * 20 - 50)
            
            size = np.random.randint(30, 80)
            
            if disaster_type == 'fire':
                color = [220, 20, 20]  
                intensity = np.random.uniform(0.7, 1.0)
            elif disaster_type == 'collapsed_building':
                color = [105, 105, 105]  
                intensity = np.random.uniform(0.8, 1.0)
            elif disaster_type == 'flooded_areas':
                color = [30, 144, 255]  
                intensity = np.random.uniform(0.6, 0.9)
            else:  
                color = [255, 255, 0]  
                intensity = np.random.uniform(0.5, 0.8)
            
            cv2.circle(aerial_image, (center_y, center_x), size, color, -1)
            
            noise = np.random.randint(-30, 30, (size*2, size*2, 3))
            x_start = max(0, center_x - size)
            x_end = min(grid_size * 20, center_x + size)
            y_start = max(0, center_y - size)
            y_end = min(grid_size * 20, center_y + size)
            
            area = aerial_image[x_start:x_end, y_start:y_end]
            noise_area = noise[:area.shape[0], :area.shape[1]]
            aerial_image[x_start:x_end, y_start:y_end] = np.clip(area + noise_area, 0, 255)
            
            disaster_locations.append({
                'type': disaster_type,
                'position': (center_x // 20, center_y // 20),
                'intensity': intensity,
                'grid_coords': (center_x // 20, center_y // 20)
            })
        
        return aerial_image.astype(np.uint8), disaster_locations
    
    def process_aerial_image_grid(self, aerial_image: np.ndarray, grid_size: int = 30) -> tuple:
        hazard_map = np.zeros((grid_size, grid_size))
        confidence_map = np.zeros((grid_size, grid_size))
        
        cell_size = aerial_image.shape[0] // grid_size
        
        for i in range(grid_size):
            for j in range(grid_size):
                x_start = i * cell_size
                x_end = (i + 1) * cell_size
                y_start = j * cell_size
                y_end = (j + 1) * cell_size
                
                cell_image = aerial_image[x_start:x_end, y_start:y_end]
                
                hazard_prob, confidence = self._classify_image_patch(cell_image)
                hazard_map[i, j] = hazard_prob
                confidence_map[i, j] = confidence
        
        return hazard_map, confidence_map
    
    def _classify_image_patch(self, image_patch: np.ndarray) -> tuple:
        if image_patch.size == 0:
            return 0.0, 0.0
        
        image_pil = Image.fromarray(image_patch)
        
        try:
            image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.classifier(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                disaster_classes = ['fire', 'collapsed_building', 'flooded_areas', 'traffic_incident']
                disaster_prob = 0.0
                max_confidence = 0.0
                
                for cls in disaster_classes:
                    if cls in self.class_to_idx:
                        prob = probabilities[0][self.class_to_idx[cls]].item()
                        disaster_prob += prob
                        max_confidence = max(max_confidence, prob)
                
                overall_confidence = torch.max(probabilities).item()
                
                return min(disaster_prob, 1.0), overall_confidence
                
        except Exception as e:
            return 0.0, 0.0
    
    def visualize_aerial_analysis(self, aerial_image: np.ndarray, hazard_map: np.ndarray, 
                                 confidence_map: np.ndarray, save_path: str = None):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(aerial_image)
        axes[0].set_title('Generated Aerial Imagery')
        axes[0].axis('off')
        
        im1 = axes[1].imshow(hazard_map, cmap='Reds', alpha=0.8)
        axes[1].set_title('Classified Disaster Risk Map')
        plt.colorbar(im1, ax=axes[1], label='Disaster Probability')
        
        im2 = axes[2].imshow(confidence_map, cmap='Blues', alpha=0.8)
        axes[2].set_title('Classification Confidence Map')
        plt.colorbar(im2, ax=axes[2], label='Confidence Level')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def demonstrate_aerial_image_processing():
    print("Aerial Image Processing with Disaster Detection")
    print("Real-time classification of simulated aerial imagery")
    print("-" * 60)
    
    processor = AerialImageProcessor()
    
    print(f"Classifier: {processor.config.model_name}")
    print(f"Classes: {list(processor.class_to_idx.keys())}")
    print("Generating aerial scene...")
    
    aerial_image, disaster_locations = processor.generate_aerial_scene(grid_size=30)
    
    print(f"Generated {len(disaster_locations)} disaster zones:")
    for i, disaster in enumerate(disaster_locations):
        print(f"  {i+1}. {disaster['type']} at grid {disaster['grid_coords']} (intensity: {disaster['intensity']:.2f})")
    
    print("\nProcessing aerial imagery through disaster classifier...")
    hazard_map, confidence_map = processor.process_aerial_image_grid(aerial_image, grid_size=30)
    
    detected_hazards = np.sum(hazard_map > 0.3)
    avg_confidence = np.mean(confidence_map)
    max_risk = np.max(hazard_map)
    
    print(f"Detection Results:")
    print(f"  High-risk areas detected: {detected_hazards}")
    print(f"  Maximum risk level: {max_risk:.3f}")
    print(f"  Average confidence: {avg_confidence:.3f}")
    
    fig = processor.visualize_aerial_analysis(aerial_image, hazard_map, confidence_map, 
                                            'aerial_image_analysis.png')
    plt.show()
    
    print("\nAerial image processing demonstration complete!")
    print("System ready for real-time UAV navigation integration.")
    
    return hazard_map, confidence_map, disaster_locations

if __name__ == "__main__":
    demonstrate_aerial_image_processing()