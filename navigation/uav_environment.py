import gym
import numpy as np
import torch
from gym import spaces
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple, Dict, Optional, List
import cv2
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TrainingConfig
from models import create_model
from disaster_dataset import get_transforms
from aerial_image_processor import AerialImageProcessor

class UAVNavigationEnv(gym.Env):
    def __init__(self, grid_size: int = 50, max_steps: int = 200, 
                 classifier_path: str = "../checkpoints/best_model.pth", 
                 cache_imagery: bool = True,
                 observation_radius: int = 2,
                 confidence_decay: float = 0.95):
        super(UAVNavigationEnv, self).__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        self.cache_imagery = cache_imagery
        self.observation_radius = observation_radius
        self.confidence_decay = confidence_decay
        self.cached_aerial_image = None
        self.observed_cells = set()
        self.last_update_step = {}
        
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(grid_size, grid_size, 4), 
            dtype=np.float32
        )
        
        self._load_disaster_classifier(classifier_path)
        self.aerial_processor = AerialImageProcessor(classifier_path)
        self._initialize_environment()
        
    def _load_disaster_classifier(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.class_to_idx = checkpoint['class_to_idx']
        
        self.classifier = create_model(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes
        )
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.to(self.device)
        self.classifier.eval()
        
        _, self.transform = get_transforms(self.config.input_size, augment=False)
        
    def _initialize_environment(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.hazard_map = np.full((self.grid_size, self.grid_size), -1.0, dtype=np.float32)
        self.confidence_map = np.full((self.grid_size, self.grid_size), -1.0, dtype=np.float32)
        self.disaster_locations = []
        self.observed_cells = set()
        self.last_update_step = {}
        
        self.uav_position = np.array([2, 2])
        self.goal_position = np.array([self.grid_size-3, self.grid_size-3])
        
        self._generate_aerial_scene()
        self._classify_local_area(self.uav_position)
        self.path_history = [self.uav_position.copy()]

    def _generate_aerial_scene(self):
        if self.cache_imagery and self.cached_aerial_image is not None:
            self.aerial_image = self.cached_aerial_image.copy()
            self.disaster_locations = self.cached_disaster_locations.copy()
            return
        
        print("Generating aerial imagery...")
        self.aerial_image, self.disaster_locations = self.aerial_processor.generate_aerial_scene(self.grid_size)
        
        if self.cache_imagery:
            self.cached_aerial_image = self.aerial_image.copy()
            self.cached_disaster_locations = self.disaster_locations.copy()
        
        print(f"Generated aerial imagery: {len(self.disaster_locations)} disaster zones")
    
    def _classify_local_area(self, position: np.ndarray):
        x, y = position
        cell_size = self.aerial_image.shape[0] // self.grid_size
        
        cells_to_classify = []
        for dx in range(-self.observation_radius, self.observation_radius + 1):
            for dy in range(-self.observation_radius, self.observation_radius + 1):
                nx = x + dx
                ny = y + dy
                
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cell_key = (nx, ny)
                    if cell_key not in self.observed_cells:
                        cells_to_classify.append((nx, ny))
        
        if cells_to_classify:
            batch_images = []
            batch_coords = []
            
            for i, j in cells_to_classify:
                x_start = i * cell_size
                x_end = (i + 1) * cell_size
                y_start = j * cell_size
                y_end = (j + 1) * cell_size
                
                cell_image = self.aerial_image[x_start:x_end, y_start:y_end]
                image_pil = Image.fromarray(cell_image)
                image_tensor = self.transform(image_pil)
                batch_images.append(image_tensor)
                batch_coords.append((i, j))
            
            if batch_images:
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
                        self.hazard_map[i, j] = min(disaster_prob, 1.0)
                        self.confidence_map[i, j] = overall_confidence
                        self.observed_cells.add((i, j))
                        self.last_update_step[(i, j)] = self.current_step
    
    def _apply_confidence_decay(self):
        for (i, j) in list(self.observed_cells):
            steps_since_update = self.current_step - self.last_update_step.get((i, j), 0)
            if steps_since_update > 0:
                decay_factor = self.confidence_decay ** steps_since_update
                self.confidence_map[i, j] = max(0.0, self.confidence_map[i, j] * decay_factor)
    
    def _get_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float32)
        
        hazard_channel = np.clip(self.hazard_map, 0, 1)
        hazard_channel[self.hazard_map < 0] = 0.0
        
        confidence_channel = np.clip(self.confidence_map, 0, 1)
        confidence_channel[self.confidence_map < 0] = 0.0
        
        obs[:, :, 0] = hazard_channel
        obs[:, :, 1] = confidence_channel
        
        obs[self.uav_position[0], self.uav_position[1], 2] = 1.0
        obs[self.goal_position[0], self.goal_position[1], 3] = 1.0
        
        return obs.astype(np.float32)
    
    def _classify_current_area(self):
        x, y = self.uav_position
        
        current_hazard = max(0.0, self.hazard_map[x, y])
        current_confidence = max(0.0, self.confidence_map[x, y])
        
        local_area = self.hazard_map[max(0, x-2):min(self.grid_size, x+3),
                                    max(0, y-2):min(self.grid_size, y+3)]
        local_area = local_area[local_area >= 0]
        
        max_local_hazard = np.max(local_area) if local_area.size > 0 else 0.0
        
        in_danger = current_hazard > 0.5 or max_local_hazard > 0.7
        
        return max(current_hazard, max_local_hazard), in_danger
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.current_step += 1
        
        self._apply_confidence_decay()
        
        actions = {
            0: [-1, 0],   # N
            1: [-1, 1],   # NE
            2: [0, 1],    # E
            3: [1, 1],    # SE
            4: [1, 0],    # S
            5: [1, -1],   # SW
            6: [0, -1],   # W
            7: [-1, -1]   # NW
        }
        
        new_position = self.uav_position + np.array(actions[action])
        new_position = np.clip(new_position, 0, self.grid_size - 1)
        
        self.uav_position = new_position
        self.path_history.append(self.uav_position.copy())
        
        self._classify_local_area(self.uav_position)
        
        reward = self._calculate_reward()
        done = self._check_terminal_conditions()
        
        hazard_level, in_danger = self._classify_current_area()
        
        info = {
            'uav_position': self.uav_position.copy(),
            'goal_position': self.goal_position.copy(),
            'hazard_level': hazard_level,
            'in_danger_zone': in_danger,
            'steps': self.current_step,
            'path_history': self.path_history.copy()
        }
        
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self) -> float:
        goal_distance = np.linalg.norm(self.uav_position - self.goal_position)
        
        if np.array_equal(self.uav_position, self.goal_position):
            return 500.0
        
        progress_reward = 100.0 / (1.0 + goal_distance)
        
        hazard_level = max(0.0, self.hazard_map[self.uav_position[0], self.uav_position[1]])
        
        if hazard_level > 0.8:
            safety_penalty = -500.0 * (hazard_level ** 3)
        elif hazard_level > 0.5:
            safety_penalty = -300.0 * (hazard_level ** 2)
        elif hazard_level > 0.2:
            safety_penalty = -100.0 * hazard_level
        else:
            safety_penalty = -10.0 * hazard_level
        
        efficiency_penalty = -0.5
        
        collision_penalty = -2000.0 if hazard_level > 0.9 else 0.0
        
        return progress_reward + safety_penalty + efficiency_penalty + collision_penalty
    
    def _check_terminal_conditions(self) -> bool:
        if np.array_equal(self.uav_position, self.goal_position):
            return True
        if self.current_step >= self.max_steps:
            return True
        hazard_level = max(0.0, self.hazard_map[self.uav_position[0], self.uav_position[1]])
        if hazard_level > 0.9:
            return True
        return False
    
    def reset(self) -> np.ndarray:
        self.current_step = 0
        self._initialize_environment()
        return self._get_observation()
    
    def render_static(self, title="UAV Navigation"):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        hazard_display = ax.imshow(self.hazard_map, cmap='Reds', alpha=0.8, origin='upper')
        
        if len(self.path_history) > 1:
            path_array = np.array(self.path_history)
            ax.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=3, alpha=0.8)
        
        ax.plot(self.uav_position[1], self.uav_position[0], 'ko', markersize=12)
        ax.plot(self.goal_position[1], self.goal_position[0], 'g*', markersize=18)
        
        ax.set_xlim(-0.5, self.grid_size-0.5)
        ax.set_ylim(-0.5, self.grid_size-0.5)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(hazard_display, ax=ax, label='Hazard Level')
        return fig
    
    def close(self):
        plt.close('all')