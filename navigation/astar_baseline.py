import numpy as np
from typing import List, Tuple, Optional
import heapq

class AStarBaseline:
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        
    def plan_path(self, start: np.ndarray, goal: np.ndarray, 
                  hazards: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        
        start_tuple = tuple(start)
        goal_tuple = tuple(goal)
        hazard_set = {tuple(h) for h in hazards}
        
        open_set = [(0, start_tuple)]
        came_from = {}
        g_score = {start_tuple: 0}
        f_score = {start_tuple: self._heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal_tuple:
                return self._reconstruct_path(came_from, current)
            
            for neighbor in self._get_neighbors(current):
                if not self._is_valid(neighbor, hazard_set):
                    continue
                    
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(
                        np.array(neighbor), goal
                    )
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def _heuristic(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        return np.linalg.norm(pos1 - pos2)
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_pos = (pos[0] + dx, pos[1] + dy)
            neighbors.append(new_pos)
        return neighbors
    
    def _is_valid(self, pos: Tuple[int, int], hazards: set) -> bool:
        return (0 <= pos[0] < self.grid_size and 
                0 <= pos[1] < self.grid_size and
                pos not in hazards)
    
    def _reconstruct_path(self, came_from: dict, 
                         current: Tuple[int, int]) -> List[np.ndarray]:
        path = [np.array(current)]
        while current in came_from:
            current = came_from[current]
            path.append(np.array(current))
        path.reverse()
        return path

class BaselineAgent:
    def __init__(self, grid_size: int):
        self.planner = AStarBaseline(grid_size)
        self.current_path = None
        self.path_index = 0
        
    def get_action(self, observation: np.ndarray) -> int:
        drone_pos = self._get_drone_position(observation)
        target_pos = self._get_target_position(observation)
        hazards = self._get_hazard_positions(observation)
        
        if (self.current_path is None or 
            self.path_index >= len(self.current_path) or
            not np.array_equal(self.current_path[-1], target_pos)):
            
            self.current_path = self.planner.plan_path(drone_pos, target_pos, hazards)
            self.path_index = 0
            
            if self.current_path is None:
                return np.random.randint(4)
        
        if self.path_index < len(self.current_path) - 1:
            next_pos = self.current_path[self.path_index + 1]
            action = self._pos_to_action(drone_pos, next_pos)
            self.path_index += 1
            return action
            
        return np.random.randint(4)
    
    def _get_drone_position(self, obs: np.ndarray) -> np.ndarray:
        drone_channel = obs[:, :, 0]
        pos = np.where(drone_channel == 1.0)
        return np.array([pos[0][0], pos[1][0]])
    
    def _get_target_position(self, obs: np.ndarray) -> np.ndarray:
        target_channel = obs[:, :, 1]
        pos = np.where(target_channel == 1.0)
        return np.array([pos[0][0], pos[1][0]])
    
    def _get_hazard_positions(self, obs: np.ndarray) -> List[np.ndarray]:
        hazard_channel = obs[:, :, 2]
        positions = np.where(hazard_channel == 1.0)
        return [np.array([x, y]) for x, y in zip(positions[0], positions[1])]
    
    def _pos_to_action(self, current: np.ndarray, next_pos: np.ndarray) -> int:
        diff = next_pos - current
        action_map = {
            (0, 1): 0,   # right
            (0, -1): 1,  # left
            (1, 0): 2,   # down
            (-1, 0): 3   # up
        }
        return action_map.get(tuple(diff), 0)

    def reset(self):
        self.current_path = None
        self.path_index = 0