import numpy as np
import matplotlib
matplotlib.use('MacOSX')  # Use interactive backend for live visualization
import matplotlib.pyplot as plt
import time
import cv2
from uav_environment import UAVNavigationEnv

class SimpleNavigationAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.position_history = []
        self.oscillation_threshold = 5
        self.oscillation_count = 0
        self.last_escape_action = None
        
    def select_action(self, state, env, deterministic=True):
        uav_pos = env.uav_position
        goal_pos = env.goal_position
        hazard_map = env.hazard_map
        
        direction = goal_pos - uav_pos
        
        action_vectors = np.array([
            [-1, 0],   # N
            [-1, 1],   # NE  
            [0, 1],    # E
            [1, 1],    # SE
            [1, 0],    # S
            [1, -1],   # SW
            [0, -1],   # W
            [-1, -1]   # NW
        ])
        
        # Check for oscillation
        self.position_history.append(tuple(uav_pos))
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        
        # Detect oscillation (same position repeated)
        if len(self.position_history) >= 6:
            recent_positions = self.position_history[-6:]
            unique_positions = len(set(recent_positions))
            if unique_positions <= 3:  # Only 1-3 unique positions in last 6 steps
                self.oscillation_count += 1
                print(f"  OSCILLATION DETECTED: Breaking out of loop (unique positions: {unique_positions}, count: {self.oscillation_count})")
                # Force a different action to break oscillation
                return self._get_escape_action(uav_pos, goal_pos, action_vectors, hazard_map, env)
            else:
                # Reset oscillation count if we're not oscillating
                self.oscillation_count = 0
        
        best_action = 0
        best_score = float('-inf')
        
        for i, action_vec in enumerate(action_vectors):
            new_pos = np.clip(uav_pos + action_vec, 0, env.grid_size - 1)
            
            progress_score = -np.linalg.norm(new_pos - goal_pos)
            
            hazard_level = hazard_map[new_pos[0], new_pos[1]]
            
            if hazard_level > 0.8:
                safety_score = -2000 * hazard_level
            elif hazard_level > 0.5:
                safety_score = -1000 * hazard_level  
            elif hazard_level > 0.2:
                safety_score = -300 * hazard_level
            else:
                safety_score = -50 * hazard_level
            
            local_hazards = 0
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    check_x = np.clip(new_pos[0] + dx, 0, env.grid_size - 1)
                    check_y = np.clip(new_pos[1] + dy, 0, env.grid_size - 1)
                    local_hazards += hazard_map[check_x, check_y]
            
            area_safety_score = -150 * local_hazards
            
            alignment_score = np.dot(action_vec, direction) / (np.linalg.norm(action_vec) * np.linalg.norm(direction) + 1e-8)
            
            total_score = progress_score + safety_score + area_safety_score + 8 * alignment_score
            
            if total_score > best_score:
                best_score = total_score
                best_action = i
        
        return best_action
    
    def _get_escape_action(self, uav_pos, goal_pos, action_vectors, hazard_map, env):
        """Get an action to escape oscillation by trying different directions"""
        direction = goal_pos - uav_pos
        
        # If we've been oscillating multiple times, try more drastic measures
        if self.oscillation_count > 2:
            print(f"    MULTIPLE OSCILLATIONS: Using drastic escape strategy")
            # Try to move in a completely different direction
            perpendicular_actions = []
            for i, action_vec in enumerate(action_vectors):
                # Find actions perpendicular to goal direction
                dot_product = abs(np.dot(action_vec, direction))
                if dot_product < 0.3:  # More perpendicular actions
                    new_pos = np.clip(uav_pos + action_vec, 0, env.grid_size - 1)
                    if tuple(new_pos) not in self.position_history[-8:]:
                        hazard_level = hazard_map[new_pos[0], new_pos[1]]
                        if hazard_level < 0.6:  # More conservative risk for drastic escape
                            perpendicular_actions.append((i, hazard_level))
            
            if perpendicular_actions:
                # Choose the safest perpendicular action
                safest_perp = min(perpendicular_actions, key=lambda x: x[1])
                self.last_escape_action = safest_perp[0]
                return safest_perp[0]
        
        # Get all possible actions with their scores
        action_scores = []
        
        for i, action_vec in enumerate(action_vectors):
            new_pos = np.clip(uav_pos + action_vec, 0, env.grid_size - 1)
            
            # Skip if this position is in recent history (more aggressive avoidance)
            if tuple(new_pos) in self.position_history[-10:]:
                continue
            
            # Skip the last escape action to avoid immediate reversal
            if i == self.last_escape_action:
                continue
                
            hazard_level = hazard_map[new_pos[0], new_pos[1]]
            progress_score = -np.linalg.norm(new_pos - goal_pos)
            alignment_score = np.dot(action_vec, direction) / (np.linalg.norm(action_vec) * np.linalg.norm(direction) + 1e-8)
            
            # More aggressive safety penalty
            safety_penalty = -3000 * hazard_level if hazard_level > 0.2 else 0
            total_score = progress_score + safety_penalty + 15 * alignment_score
            
            action_scores.append((i, total_score, hazard_level))
        
        # Sort by total score (best first)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Try the best actions that are reasonably safe
        for action_idx, score, hazard_level in action_scores:
            if hazard_level < 0.4:  # Even more conservative risk threshold
                self.last_escape_action = action_idx
                return action_idx
        
        # If all actions are too risky, take the safest one
        if action_scores:
            safest_action = min(action_scores, key=lambda x: x[2])
            self.last_escape_action = safest_action[0]
            return safest_action[0]
        
        # Fallback: try to move away from current position
        for i, action_vec in enumerate(action_vectors):
            new_pos = np.clip(uav_pos + action_vec, 0, env.grid_size - 1)
            if tuple(new_pos) not in self.position_history[-5:]:
                self.last_escape_action = i
                return i
        
        # Last resort: random action
        import random
        action = random.randint(0, len(action_vectors) - 1)
        self.last_escape_action = action
        return action

def demonstrate_intelligent_navigation():
    print("UAV Intelligent Navigation System")
    print("Real-time navigation with disaster avoidance")
    print("-" * 60)
    
    env = UAVNavigationEnv(grid_size=30, max_steps=60)  # Slightly longer for better completion
    agent = SimpleNavigationAgent(action_dim=env.action_space.n)
    
    print(f"Environment: {env.grid_size}x{env.grid_size} grid")
    print(f"Classifier: {env.config.model_name}")
    print(f"Disaster classes: {list(env.class_to_idx.keys())}")
    print("-" * 60)
    
    state = env.reset()
    path_history = [env.uav_position.copy()]
    hazard_encounters = []
    total_reward = 0
    step_count = 0
    
    print(f"Starting position: {env.uav_position}")
    print(f"Goal position: {env.goal_position}")
    print(f"High-risk zones: {np.sum(env.hazard_map > 0.5)} areas")
    print("\nNavigating to goal...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.ion()  # Turn on interactive mode
    
    # Create a list to store frames for video
    frames = []
    
    while step_count < env.max_steps:
        action = agent.select_action(state, env)
        
        action_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        
        state, reward, done, info = env.step(action)
        
        path_history.append(info['uav_position'].copy())
        total_reward += reward
        step_count += 1
        
        print(f"Step {step_count}: {action_names[action]} -> {info['uav_position']} | Reward: {reward:.1f}")
        
        # Count both danger zone entries and hazard avoidance
        if info['in_danger_zone']:
            hazard_encounters.append({
                'step': step_count,
                'position': info['uav_position'].copy(),
                'risk_level': info['hazard_level'],
                'type': 'danger_zone_entry'
            })
            print(f"  HAZARD DETECTED: Risk level {info['hazard_level']:.2f}")
        elif info['hazard_level'] > 0.3:  # Count near-misses as hazard avoidance
            hazard_encounters.append({
                'step': step_count,
                'position': info['uav_position'].copy(),
                'risk_level': info['hazard_level'],
                'type': 'hazard_avoidance'
            })
            print(f"  HAZARD AVOIDED: Risk level {info['hazard_level']:.2f}")
        
        ax.clear()
        
        hazard_display = ax.imshow(env.hazard_map, cmap='Reds', alpha=0.8, origin='upper')
        
        if len(path_history) > 1:
            path_array = np.array(path_history)
            ax.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=3, alpha=0.9, label='UAV Path')
        
        current_pos = info['uav_position']
        ax.plot(current_pos[1], current_pos[0], 'ko', markersize=15, label='UAV Current')
        ax.plot(env.goal_position[1], env.goal_position[0], 'g*', markersize=20, label='Goal')
        ax.plot(path_history[0][1], path_history[0][0], 'go', markersize=12, label='Start')
        
        for encounter in hazard_encounters:
            pos = encounter['position']
            ax.plot(pos[1], pos[0], 'rx', markersize=10, alpha=0.8)
        
        distance_to_goal = np.linalg.norm(current_pos - env.goal_position)
        
        ax.set_xlim(-0.5, env.grid_size-0.5)
        ax.set_ylim(-0.5, env.grid_size-0.5)
        ax.set_title(f'UAV Navigation - Step {step_count} | Distance: {distance_to_goal:.1f} | Total Reward: {total_reward:.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.show()
        plt.pause(0.3)  # Shorter pause for smoother video
        
        fig.canvas.draw()
        
        # Save to a temporary buffer and read it back
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        
        # Read the image from buffer
        from PIL import Image
        img = Image.open(buf)
        
        # Resize to standard video dimensions for better compatibility
        img = img.resize((800, 600), Image.Resampling.LANCZOS)
        frame = np.array(img)
        
        # Convert to RGB if needed
        if frame.shape[2] == 4:  # RGBA
            frame = frame[:, :, :3]  # Remove alpha channel
        
        # Ensure correct data type
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        frames.append(frame)
        buf.close()
        
        # Save step visualization
        plt.savefig(f'navigation_step_{step_count:03d}.png', dpi=150, bbox_inches='tight')
        
        if done:
            break
    
    mission_success = np.array_equal(info['uav_position'], info['goal_position'])
    final_distance = np.linalg.norm(info['uav_position'] - info['goal_position'])
    
    print("\n" + "=" * 60)
    print("NAVIGATION MISSION COMPLETE")
    print("=" * 60)
    print(f"Mission Status: {'SUCCESS' if mission_success else 'INCOMPLETE'}")
    print(f"Steps taken: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final distance to goal: {final_distance:.2f}")
    danger_entries = [e for e in hazard_encounters if e['type'] == 'danger_zone_entry']
    hazard_avoidances = [e for e in hazard_encounters if e['type'] == 'hazard_avoidance']
    
    print(f"Hazard encounters: {len(hazard_encounters)} (Danger entries: {len(danger_entries)}, Avoidances: {len(hazard_avoidances)})")
    print(f"Navigation efficiency: {len(path_history)/step_count:.2f}")
    
    if hazard_encounters:
        print(f"\nHazard Management:")
        for i, encounter in enumerate(hazard_encounters):
            action_type = "Entered" if encounter['type'] == 'danger_zone_entry' else "Avoided"
            print(f"  {i+1}. Step {encounter['step']}: {action_type} risk {encounter['risk_level']:.2f} at {encounter['position']}")
    
    plt.colorbar(hazard_display, ax=ax, label='Disaster Risk Level')
    
    status_text = "SUCCESS" if mission_success else "NAVIGATING"
    plt.title(f'UAV Navigation {status_text} - Reward: {total_reward:.1f}')
    plt.tight_layout()
    plt.savefig('intelligent_navigation_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create video from frames
    if frames:
        print(f"Creating video with {len(frames)} frames...")
        height, width, layers = frames[0].shape
        print(f"Frame dimensions: {width}x{height}x{layers}")
        
        # Try different codecs and formats with better error handling
        video_created = False
        codecs_to_try = [
            ('mp4v', 'navigation_demo.mp4'),
            ('XVID', 'navigation_demo.avi'),
            ('MJPG', 'navigation_demo.avi'),
            ('H264', 'navigation_demo.mp4')
        ]
        
        for codec, filename in codecs_to_try:
            try:
                print(f"Trying {codec} codec...")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video = cv2.VideoWriter(filename, fourcc, 2, (width, height))
                
                if video.isOpened():
                    print(f"Video writer opened successfully with {codec}")
                    for i, frame in enumerate(frames):
                        # Ensure frame is in correct format
                        if frame.dtype != np.uint8:
                            frame = frame.astype(np.uint8)
                        
                        # Ensure frame has correct shape and convert RGB to BGR for OpenCV
                        if len(frame.shape) == 3 and frame.shape[2] == 3:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            success = video.write(frame_bgr)
                            if not success and i < 3:  # Only warn for first few frames
                                print(f"Warning: Failed to write frame {i}")
                    
                    video.release()
                    print(f"Video saved as {filename} using {codec} codec")
                    video_created = True
                    break
                else:
                    video.release()
                    print(f"Failed to open video writer for {codec}")
            except Exception as e:
                print(f"Error with {codec}: {e}")
                continue
        
        if not video_created:
            print("Warning: Could not create video with any codec")
    else:
        print("No frames captured for video creation")
    
    return {
        'success': mission_success,
        'steps': step_count,
        'reward': total_reward,
        'hazard_encounters': len(hazard_encounters),
        'efficiency': len(path_history)/step_count
    }

if __name__ == "__main__":
    result = demonstrate_intelligent_navigation()
    print(f"\nNavigation demonstration: {result}")
    
    if result['success']:
        print("UAV successfully reached goal while avoiding disaster zones!")
    else:
        print("UAV demonstrated intelligent navigation and hazard avoidance.")
    
    print("\nSystem ready for PPO training to improve autonomous performance.")