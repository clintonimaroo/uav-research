import gym
import numpy as np
from gym import spaces
from typing import Tuple, Optional, List

class DroneNavigationEnv(gym.Env):
    def __init__(self, grid_size: int = 20, max_steps: int = 100):
        super(DroneNavigationEnv, self).__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.float32
        )
        
        self.drone_pos = None
        self.target_pos = None
        self.hazard_zones = []
        self.grid = None
        
    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        self.drone_pos = self._random_position()
        self.target_pos = self._random_position()
        while np.array_equal(self.drone_pos, self.target_pos):
            self.target_pos = self._random_position()
        
        self._generate_hazards()
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.current_step += 1
        
        old_pos = self.drone_pos.copy()
        new_pos = self._apply_action(action)
        
        if self._is_valid_position(new_pos):
            self.drone_pos = new_pos
        
        reward = self._calculate_reward(old_pos)
        done = self._is_terminal()
        
        return self._get_observation(), reward, done, {}
    
    def _apply_action(self, action: int) -> np.ndarray:
        moves = {0: [0, 1], 1: [0, -1], 2: [1, 0], 3: [-1, 0]}
        return self.drone_pos + np.array(moves[action])
    
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        return (0 <= pos[0] < self.grid_size and 
                0 <= pos[1] < self.grid_size and
                not self._is_hazard(pos))
    
    def _is_hazard(self, pos: np.ndarray) -> bool:
        return any(np.array_equal(pos, h) for h in self.hazard_zones)
    
    def _calculate_reward(self, old_pos: np.ndarray) -> float:
        if np.array_equal(self.drone_pos, self.target_pos):
            return 100.0
        
        if self._is_hazard(self.drone_pos):
            return -50.0
        
        old_dist = np.linalg.norm(old_pos - self.target_pos)
        new_dist = np.linalg.norm(self.drone_pos - self.target_pos)
        
        return (old_dist - new_dist) * 10 - 1
    
    def _is_terminal(self) -> bool:
        return (np.array_equal(self.drone_pos, self.target_pos) or
                self.current_step >= self.max_steps or
                self._is_hazard(self.drone_pos))
    
    def _get_observation(self) -> np.ndarray:
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        obs[self.drone_pos[0], self.drone_pos[1], 0] = 1.0
        obs[self.target_pos[0], self.target_pos[1], 1] = 1.0
        
        for hazard in self.hazard_zones:
            obs[hazard[0], hazard[1], 2] = 1.0
        
        return obs
    
    def _random_position(self) -> np.ndarray:
        return np.array([
            np.random.randint(0, self.grid_size),
            np.random.randint(0, self.grid_size)
        ])
    
    def _generate_hazards(self):
        num_hazards = np.random.randint(3, 8)
        self.hazard_zones = []
        
        for _ in range(num_hazards):
            pos = self._random_position()
            while (np.array_equal(pos, self.drone_pos) or 
                   np.array_equal(pos, self.target_pos) or
                   any(np.array_equal(pos, h) for h in self.hazard_zones)):
                pos = self._random_position()
            self.hazard_zones.append(pos)

    def render(self, mode='human'):
        if mode == 'human':
            grid_display = np.zeros((self.grid_size, self.grid_size))
            
            for hazard in self.hazard_zones:
                grid_display[hazard[0], hazard[1]] = -1
            
            grid_display[self.target_pos[0], self.target_pos[1]] = 2
            grid_display[self.drone_pos[0], self.drone_pos[1]] = 1
            
            print(grid_display)