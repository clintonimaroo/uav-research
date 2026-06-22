import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
from typing import Optional, Any, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import create_model
from disaster_dataset import get_transforms

class AerialImageProcessor:
    def __init__(
        self,
        classifier_path: str = "../checkpoints/best_model.pth",
        *,
        shared_classifier: Optional[torch.nn.Module] = None,
        shared_transform: Any = None,
        shared_device: Optional[torch.device] = None,
        shared_class_to_idx: Optional[Dict[str, int]] = None,
        shared_config: Any = None,
    ):
        if shared_classifier is not None:
            self.device = shared_device or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self.classifier = shared_classifier
            self.transform = shared_transform
            self.class_to_idx = shared_class_to_idx or {}
            self.config = shared_config
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            self._load_classifier(classifier_path)
        
    def _load_classifier(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.config = checkpoint['config']
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        state_dict = checkpoint['model_state_dict']
        del checkpoint

        self.classifier = create_model(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes
        )
        self.classifier.load_state_dict(state_dict)
        del state_dict
        self.classifier.to(self.device)
        self.classifier.eval()
        
        _, self.transform = get_transforms(self.config.input_size, augment=False)
        
    def _rng_int(self, rng, low: int, high: int) -> int:
        if hasattr(rng, "integers"):
            return int(rng.integers(low, high))
        return int(rng.randint(low, high))

    def _rng_uniform(self, rng, low: float, high: float) -> float:
        return float(rng.uniform(low, high))

    def _rng_choice(self, rng, values):
        if hasattr(rng, "choice"):
            return str(rng.choice(values))
        return str(values[self._rng_int(rng, 0, len(values))])

    def _profile_spec(self, scene_profile: str):
        profiles = {
            "fire_light": {
                "types": ["fire"],
                "count": (1, 3),
                "size": (18, 38),
                "intensity": (0.55, 0.75),
                "path_fire": True,
            },
            "fire_moderate": {
                "types": ["fire"],
                "count": (3, 5),
                "size": (24, 54),
                "intensity": (0.65, 0.9),
                "path_fire": True,
            },
            "fire_dense": {
                "types": ["fire"],
                "count": (6, 9),
                "size": (32, 76),
                "intensity": (0.75, 1.0),
            },
            "mixed": {
                "types": ["fire", "collapsed_building", "flooded_areas", "traffic_incident"],
                "count": (5, 12),
                "size": (30, 80),
                "intensity": None,
            },
        }
        return profiles.get(scene_profile, profiles["mixed"])

    def generate_aerial_scene(self, grid_size: int = 30, cell_px: int = 20, scene_profile: str = "mixed", rng=None):
        px = max(4, int(cell_px))
        side = grid_size * px
        margin = max(3, min(50, px * 2))
        rng = rng or np.random.default_rng()
        profile = self._profile_spec(scene_profile)
        aerial_image = np.zeros((side, side, 3), dtype=np.uint8)
        
        aerial_image[:, :] = [34, 139, 34]
        
        num_disasters = self._rng_int(rng, profile["count"][0], profile["count"][1])
        disaster_locations = []

        def add_disaster(disaster_type: str, center_x: int = None, center_y: int = None):
            if center_x is None:
                center_x = self._rng_int(rng, margin, side - margin)
            if center_y is None:
                center_y = self._rng_int(rng, margin, side - margin)
            size = self._rng_int(rng, profile["size"][0], profile["size"][1])
            if disaster_type == 'fire':
                color = [220, 20, 20]  
                if profile["intensity"] is None:
                    intensity = self._rng_uniform(rng, 0.7, 1.0)
                else:
                    intensity = self._rng_uniform(rng, profile["intensity"][0], profile["intensity"][1])
            elif disaster_type == 'collapsed_building':
                color = [105, 105, 105]  
                intensity = self._rng_uniform(rng, 0.8, 1.0)
            elif disaster_type == 'flooded_areas':
                color = [30, 144, 255]  
                intensity = self._rng_uniform(rng, 0.6, 0.9)
            else:  
                color = [255, 255, 0]  
                intensity = self._rng_uniform(rng, 0.5, 0.8)

            cv2.circle(aerial_image, (center_y, center_x), size, color, -1)

            if hasattr(rng, "integers"):
                noise = rng.integers(-30, 30, (size*2, size*2, 3))
            else:
                noise = rng.randint(-30, 30, (size*2, size*2, 3))
            x_start = max(0, center_x - size)
            x_end = min(side, center_x + size)
            y_start = max(0, center_y - size)
            y_end = min(side, center_y + size)
            
            area = aerial_image[x_start:x_end, y_start:y_end]
            noise_area = noise[:area.shape[0], :area.shape[1]]
            aerial_image[x_start:x_end, y_start:y_end] = np.clip(area + noise_area, 0, 255)
            
            disaster_locations.append({
                'type': disaster_type,
                'position': (center_x // px, center_y // px),
                'intensity': intensity,
                'grid_coords': (center_x // px, center_y // px)
            })

        forced_count = 0
        if profile.get("path_fire"):
            start_cell = 2
            goal_cell = grid_size - 3
            path_fraction = self._rng_uniform(rng, 0.34, 0.62)
            path_cell = start_cell + path_fraction * (goal_cell - start_cell)
            jitter_cells = self._rng_uniform(rng, -1.25, 1.25)
            center_x = int(np.clip(round((path_cell + jitter_cells) * px), margin, side - margin - 1))
            center_y = int(np.clip(round((path_cell - jitter_cells) * px), margin, side - margin - 1))
            add_disaster("fire", center_x=center_x, center_y=center_y)
            forced_count = 1
        
        for _ in range(max(0, num_disasters - forced_count)):
            add_disaster(self._rng_choice(rng, profile["types"]))
        
        return aerial_image.astype(np.uint8), disaster_locations
    
    def process_aerial_image_grid(self, aerial_image: np.ndarray, grid_size: int = 30, batch_size: int = 256) -> tuple:
        hazard_map = np.zeros((grid_size, grid_size))
        confidence_map = np.zeros((grid_size, grid_size))
        
        cell_size = aerial_image.shape[0] // grid_size

        def flush_batch(batch_images, batch_coords):
            if not batch_images:
                return
            batch_tensor = torch.stack(batch_images).to(self.device)

            with torch.no_grad():
                outputs = self.classifier(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)

                disaster_classes = ['fire', 'collapsed_building', 'flooded_areas', 'traffic_incident']

                for idx, (i, j) in enumerate(batch_coords):
                    disaster_prob = 0.0

                    for cls in disaster_classes:
                        if cls in self.class_to_idx:
                            prob = probabilities[idx][self.class_to_idx[cls]].item()
                            disaster_prob += prob

                    overall_confidence = torch.max(probabilities[idx]).item()
                    hazard_map[i, j] = min(disaster_prob, 1.0)
                    confidence_map[i, j] = overall_confidence

            del batch_tensor
            del outputs, probabilities

        batch_images = []
        batch_coords = []

        for i in range(grid_size):
            for j in range(grid_size):
                x_start = i * cell_size
                x_end = (i + 1) * cell_size
                y_start = j * cell_size
                y_end = (j + 1) * cell_size
                
                cell_image = aerial_image[x_start:x_end, y_start:y_end]
                image_pil = Image.fromarray(cell_image)
                image_tensor = self.transform(image_pil)
                batch_images.append(image_tensor)
                batch_coords.append((i, j))

                if len(batch_images) >= batch_size:
                    flush_batch(batch_images, batch_coords)
                    batch_images = []
                    batch_coords = []

        flush_batch(batch_images, batch_coords)
        
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
