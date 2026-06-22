import gym
import json
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
                 confidence_decay: float = 0.95,
                 termination_threshold: float = 0.9,
                 perception_noise_std: float = 0.0,
                 lightweight_info: bool = False,
                 aerial_cell_px: int = 20,
                 quiet_scene: bool = False,
                 scene_profile: str = "mixed",
                 scene_seed: Optional[int] = None,
                 fixed_map_path: Optional[str] = None):
        super(UAVNavigationEnv, self).__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.aerial_cell_px = max(4, int(aerial_cell_px))
        self.current_step = 0
        self.cache_imagery = cache_imagery
        self.quiet_scene = quiet_scene
        self.observation_radius = observation_radius
        self.confidence_decay = confidence_decay
        self.termination_threshold = termination_threshold
        self.perception_noise_std = perception_noise_std
        self.lightweight_info = lightweight_info
        self.scene_profile = scene_profile
        self.scene_seed = scene_seed
        self.fixed_map_path = fixed_map_path
        self.fixed_map_metadata = None
        self.fixed_aerial_image = None
        self.fixed_disaster_locations = None
        self.fixed_gt_hazard_map = None
        self.fixed_classifier_hazard_map = None
        self.fixed_confidence_map = None
        self.cached_aerial_image = None
        self.cached_scene_seed = None
        self.observed_cells = set()
        self.last_update_step = {}
        self.cell_classification_cache = {}
        self.previous_distance = None
        
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(grid_size, grid_size, 4), 
            dtype=np.float32
        )
        
        self._load_disaster_classifier(classifier_path)
        if self.fixed_map_path:
            self._load_fixed_map_package(self.fixed_map_path)
        self.aerial_processor = AerialImageProcessor(
            classifier_path,
            shared_classifier=self.classifier,
            shared_transform=self.transform,
            shared_device=self.device,
            shared_class_to_idx=self.class_to_idx,
            shared_config=self.config,
        )
        self._initialize_environment()

    def _load_fixed_map_package(self, fixed_map_path: str):
        map_path = os.path.abspath(fixed_map_path)
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Fixed map package not found: {map_path}")

        package = np.load(map_path, allow_pickle=False)
        self.fixed_aerial_image = package["aerial_image"].astype(np.uint8)
        self.fixed_gt_hazard_map = package["gt_hazard_map"].astype(np.float32)
        self.fixed_classifier_hazard_map = package["classifier_hazard_map"].astype(np.float32)
        self.fixed_confidence_map = package["confidence_map"].astype(np.float32)

        disaster_json = package["disaster_locations_json"].item()
        self.fixed_disaster_locations = json.loads(disaster_json)

        metadata_path = os.path.splitext(map_path)[0] + ".json"
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as handle:
                metadata = json.load(handle)
        metadata["map_package_path"] = map_path
        metadata["metadata_path"] = metadata_path if os.path.exists(metadata_path) else ""
        self.fixed_map_metadata = metadata

        expected = (self.grid_size, self.grid_size)
        for name, array in [
            ("gt_hazard_map", self.fixed_gt_hazard_map),
            ("classifier_hazard_map", self.fixed_classifier_hazard_map),
            ("confidence_map", self.fixed_confidence_map),
        ]:
            if array.shape != expected:
                raise ValueError(
                    f"Fixed map {name} has shape {array.shape}, expected {expected}"
                )

    def map_identity(self) -> Dict:
        if self.fixed_map_metadata is not None:
            return dict(self.fixed_map_metadata)
        return {
            "map_label": self.scene_profile,
            "scene_profile": self.scene_profile,
            "scene_seed": self.scene_seed,
            "map_package_path": "",
            "map_source": "generated_at_runtime",
        }
        
    def _load_disaster_classifier(self, model_path: str):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.config = checkpoint['config']
        self.class_to_idx = checkpoint['class_to_idx']
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
        
    def _initialize_environment(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.hazard_map = np.full((self.grid_size, self.grid_size), -1.0, dtype=np.float32)
        self.confidence_map = np.full((self.grid_size, self.grid_size), -1.0, dtype=np.float32)
        self.gt_hazard_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        self.disaster_locations = []
        self.observed_cells = set()
        self.last_update_step = {}
        
        self.uav_position = np.array([2, 2])
        self.goal_position = np.array([self.grid_size-3, self.grid_size-3])
        self._max_distance = np.linalg.norm(self.goal_position - self.uav_position)
        
        self._generate_aerial_scene()
        self._compute_ground_truth_hazards()
        self._classify_local_area(self.uav_position)
        self.path_history = [self.uav_position.copy()]

    def _compute_ground_truth_hazards(self):
        if self.fixed_gt_hazard_map is not None:
            self.gt_hazard_map = self.fixed_gt_hazard_map.copy()
            return

        self.gt_hazard_map.fill(0.0)
        
        for disaster in self.disaster_locations:
            center_x, center_y = disaster['grid_coords']
            intensity = disaster['intensity']
            radius = int(np.clip(intensity * 4, 1, 4))
            
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    nx = center_x + i
                    ny = center_y + j
                    
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        distance = np.sqrt(i**2 + j**2)
                        if distance <= radius:
                            falloff = max(0.0, 1.0 - (distance / (radius + 1))**2)
                            hazard_val = intensity * falloff
                            self.gt_hazard_map[nx, ny] = max(self.gt_hazard_map[nx, ny], hazard_val)

    def _generate_aerial_scene(self):
        if self.fixed_aerial_image is not None:
            self.aerial_image = self.fixed_aerial_image.copy()
            self.disaster_locations = [dict(item) for item in self.fixed_disaster_locations]
            self.cell_classification_cache = {}
            if not self.quiet_scene:
                label = self.fixed_map_metadata.get("map_label", "fixed map") if self.fixed_map_metadata else "fixed map"
                print(f"Loaded fixed map package: {label}")
            return

        if self.cached_aerial_image is not None and self.cached_scene_seed == self.scene_seed:
            self.aerial_image = self.cached_aerial_image
            self.disaster_locations = self.cached_disaster_locations
            return
        
        if not self.quiet_scene:
            print("Generating aerial imagery...")
        rng = np.random.default_rng(self.scene_seed)
        self.aerial_image, self.disaster_locations = self.aerial_processor.generate_aerial_scene(
            self.grid_size, cell_px=self.aerial_cell_px, scene_profile=self.scene_profile, rng=rng
        )
        
        self.cell_classification_cache = {}
        if self.cache_imagery or self.scene_seed is not None:
            self.cached_aerial_image = self.aerial_image
            self.cached_disaster_locations = self.disaster_locations
            self.cached_scene_seed = self.scene_seed
        
        if not self.quiet_scene:
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

        if self.fixed_classifier_hazard_map is not None:
            for i, j in cells_to_classify:
                self.hazard_map[i, j] = float(self.fixed_classifier_hazard_map[i, j])
                self.confidence_map[i, j] = float(self.fixed_confidence_map[i, j])
                self.observed_cells.add((i, j))
                self.last_update_step[(i, j)] = self.current_step
            return
        
        if cells_to_classify:
            batch_images = []
            batch_coords = []
            
            for i, j in cells_to_classify:
                cached = self.cell_classification_cache.get((i, j))
                if cached is not None:
                    self.hazard_map[i, j] = cached["hazard"]
                    self.confidence_map[i, j] = cached["confidence"]
                    self.observed_cells.add((i, j))
                    self.last_update_step[(i, j)] = self.current_step
                    continue
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
                
                with torch.inference_mode():
                    outputs = self.classifier(batch_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    
                    if self.perception_noise_std > 0:
                        noise = torch.randn_like(probabilities) * self.perception_noise_std
                        probabilities = torch.clamp(probabilities + noise, 0.0, 1.0)
                        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
                    
                    disaster_classes = ['fire', 'collapsed_building', 'flooded_areas', 'traffic_incident']
                    
                    for idx, (i, j) in enumerate(batch_coords):
                        disaster_prob = 0.0
                        
                        for cls in disaster_classes:
                            if cls in self.class_to_idx:
                                prob = probabilities[idx][self.class_to_idx[cls]].item()
                                disaster_prob += prob
                        
                        overall_confidence = torch.max(probabilities[idx]).item()
                        hazard_value = min(disaster_prob, 1.0)
                        self.hazard_map[i, j] = hazard_value
                        self.confidence_map[i, j] = overall_confidence
                        self.cell_classification_cache[(i, j)] = {
                            "hazard": hazard_value,
                            "confidence": overall_confidence,
                        }
                        self.observed_cells.add((i, j))
                        self.last_update_step[(i, j)] = self.current_step
                    del outputs, probabilities
                del batch_tensor
    
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

        if not hasattr(self, '_coord_grid'):
            rows = np.arange(self.grid_size, dtype=np.float32)
            self._coord_grid = np.stack(np.meshgrid(rows, rows, indexing='ij'))

        dy = self._coord_grid[0] - self.uav_position[0]
        dx = self._coord_grid[1] - self.uav_position[1]
        uav_dist = np.sqrt(dy * dy + dx * dx)
        obs[:, :, 2] = 1.0 - np.clip(uav_dist / self._max_distance, 0, 1)

        dy_g = self._coord_grid[0] - self.goal_position[0]
        dx_g = self._coord_grid[1] - self.goal_position[1]
        goal_dist = np.sqrt(dy_g * dy_g + dx_g * dx_g)
        obs[:, :, 3] = 1.0 - np.clip(goal_dist / self._max_distance, 0, 1)

        return obs
    
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
            0: [-1, 0],   
            1: [-1, 1],   
            2: [0, 1],    
            3: [1, 1],    
            4: [1, 0],    
            5: [1, -1],   
            6: [0, -1],   
            7: [-1, -1]   
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
            'gt_hazard_level': float(self.gt_hazard_map[self.uav_position[0], self.uav_position[1]]),
            'in_danger_zone': in_danger,
            'steps': self.current_step,
        }
        if not self.lightweight_info:
            info['path_history'] = self.path_history.copy()
        
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self) -> float:
        R_GOAL = 100.0
        R_STEP = -0.2
        R_PROGRESS_SCALE = 5.0
        R_HAZARD_MAX = -10.0
        R_COLLISION = -100.0

        goal_distance = np.linalg.norm(self.uav_position - self.goal_position)

        if np.array_equal(self.uav_position, self.goal_position):
            self.previous_distance = None
            return R_GOAL

        progress_reward = 0.0
        if self.previous_distance is not None:
            distance_change = self.previous_distance - goal_distance
            progress_reward = R_PROGRESS_SCALE * distance_change

        self.previous_distance = goal_distance

        proximity_bonus = max(0.0, 1.0 - goal_distance / self._max_distance) * 0.5

        hazard_level = max(0.0, self.hazard_map[self.uav_position[0], self.uav_position[1]])
        hazard_penalty = R_HAZARD_MAX * hazard_level

        collision_penalty = R_COLLISION if hazard_level > self.termination_threshold else 0.0

        return progress_reward + R_STEP + proximity_bonus + hazard_penalty + collision_penalty
    
    def _check_terminal_conditions(self) -> bool:
        if np.array_equal(self.uav_position, self.goal_position):
            return True
        if self.current_step >= self.max_steps:
            return True
        hazard_level = max(0.0, self.hazard_map[self.uav_position[0], self.uav_position[1]])
        if hazard_level > self.termination_threshold:
            return True
        return False
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> np.ndarray:
        if seed is not None and self.fixed_aerial_image is None:
            self.scene_seed = int(seed)
        self.current_step = 0
        self.previous_distance = None
        self._initialize_environment()
        self.previous_distance = np.linalg.norm(self.uav_position - self.goal_position)
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
