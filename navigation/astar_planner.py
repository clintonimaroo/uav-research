import heapq
import numpy as np


class AStarPlanner:
    def __init__(self, grid_size: int, diag: bool = True):
        self.grid_size = grid_size
        self.diag = diag

    def _neighbors(self, x: int, y: int):
        if self.diag:
            steps = [
                (-1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
                (1, 0),
                (1, -1),
                (0, -1),
                (-1, -1),
            ]
        else:
            steps = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        for dx, dy in steps:
            nx = max(0, min(self.grid_size - 1, x + dx))
            ny = max(0, min(self.grid_size - 1, y + dy))
            yield nx, ny

    def _heuristic(self, a: np.ndarray, b: np.ndarray) -> float:
        dx = abs(int(a[0]) - int(b[0]))
        dy = abs(int(a[1]) - int(b[1]))
        return max(dx, dy) if self.diag else dx + dy

    def plan(self, hazard_map: np.ndarray, start: np.ndarray, goal: np.ndarray):
        h, w = hazard_map.shape
        assert h == self.grid_size and w == self.grid_size

        start_t = (int(start[0]), int(start[1]))
        goal_t = (int(goal[0]), int(goal[1]))

        open_heap = []
        heapq.heappush(open_heap, (0.0, start_t))

        came_from = {}
        g_score = {start_t: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal_t:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return np.array(path, dtype=int)

            cx, cy = current
            for nx, ny in self._neighbors(cx, cy):
                step_cost = 1.0 + 50.0 * float(hazard_map[nx, ny])
                tentative_g = g_score[current] + step_cost

                neighbor = (nx, ny)
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(np.array([nx, ny]), np.array(goal_t))
                    heapq.heappush(open_heap, (f, neighbor))

        return np.array([start_t], dtype=int)

    @staticmethod
    def action_from_move(prev_pos: np.ndarray, next_pos: np.ndarray) -> int:
        dx = int(next_pos[0]) - int(prev_pos[0])
        dy = int(next_pos[1]) - int(prev_pos[1])
        mapping = {
            (-1, 0): 0,
            (-1, 1): 1,
            (0, 1): 2,
            (1, 1): 3,
            (1, 0): 4,
            (1, -1): 5,
            (0, -1): 6,
            (-1, -1): 7,
        }
        return mapping.get((dx, dy), 0)


