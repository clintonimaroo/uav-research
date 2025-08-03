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

class UAVNavigationEnv(gym.Env):
    def __init__(self, grid_size: int = 50, max_steps: int = 200, 
                 classifier_path: str = "../checkpoints/best_model.pth"):
        super(UAVNavigationEnv, self).__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(grid_size, grid_size, 4), 
            dtype=np.float32
        )
        
        self._load_disaster_classifier(classifier_path)
        self._initialize_environment()
        
    def _load_disaster_classifier(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.config = checkpoint['config']
        self.class_to_idx = checkpoint['class_to_idx']
        
        self.classifier = create_model(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes
        )
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()
        
        _, self.transform = get_transforms(self.config.input_size, augment=False)
        
    def _initialize_environment(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.hazard_map = np.zeros((self.grid_size, self.grid_size))
        self.confidence_map = np.zeros((self.grid_size, self.grid_size))
        
        self.uav_position = np.array([2, 2])
        self.goal_position = np.array([self.grid_size-3, self.grid_size-3])
        
        self._generate_disaster_zones()
        self.path_history = [self.uav_position.copy()]
        
    def _generate_disaster_zones(self):
        disaster_types = ['fire', 'collapsed_building', 'flooded_areas', 'traffic_incident']
        num_zones = np.random.randint(8, 15)
        
        for _ in range(num_zones):
            center = np.random.randint(10, self.grid_size-10, 2)
            
            if np.linalg.norm(center - self.goal_position) < 8:
                continue
            if np.linalg.norm(center - self.uav_position) < 8:
                continue
                
            zone_size = np.random.randint(3, 8)
            disaster_type = np.random.choice(disaster_types)
            intensity = np.random.uniform(0.6, 1.0)
            
            for dx in range(-zone_size, zone_size+1):
                for dy in range(-zone_size, zone_size+1):
                    x, y = center[0] + dx, center[1] + dy
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance <= zone_size:
                            fade_factor = max(0, 1 - distance / zone_size)
                            self.hazard_map[x, y] = min(1.0, intensity * fade_factor)
                            self.confidence_map[x, y] = np.random.uniform(0.7, 0.95)
    
    def _get_observation(self):
        obs = np.zeros((self.grid_size, self.grid_size, 4))
        
        obs[:, :, 0] = self.hazard_map
        obs[:, :, 1] = self.confidence_map
        
        obs[self.uav_position[0], self.uav_position[1], 2] = 1.0
        obs[self.goal_position[0], self.goal_position[1], 3] = 1.0
        
        return obs.astype(np.float32)
    
    def _classify_current_area(self):
        x, y = self.uav_position
        local_area = self.hazard_map[max(0, x-2):min(self.grid_size, x+3),
                                    max(0, y-2):min(self.grid_size, y+3)]
        
        if np.any(local_area > 0.5):
            return np.max(local_area), True
        return 0.0, False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.current_step += 1
        
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
        
        hazard_level = self.hazard_map[self.uav_position[0], self.uav_position[1]]
        safety_penalty = -200.0 * (hazard_level ** 2)
        
        efficiency_penalty = -0.5
        
        collision_penalty = -1000.0 if hazard_level > 0.9 else 0.0
        
        return progress_reward + safety_penalty + efficiency_penalty + collision_penalty
    
    def _check_terminal_conditions(self) -> bool:
        if np.array_equal(self.uav_position, self.goal_position):
            return True
        if self.current_step >= self.max_steps:
            return True
        if self.hazard_map[self.uav_position[0], self.uav_position[1]] > 0.9:
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