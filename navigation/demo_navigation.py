import numpy as np
import matplotlib.pyplot as plt
import time
from uav_environment import UAVNavigationEnv
from ppo_navigation import PPONavigationAgent

def demonstrate_real_time_navigation():
    print("UAV Disaster-Aware Navigation System")
    print("Real-time demonstration with classifier integration")
    print("-" * 60)
    
    env = UAVNavigationEnv(grid_size=40, max_steps=150)
    agent = PPONavigationAgent(
        input_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        learning_rate=1e-4
    )
    
    print(f"Environment: {env.grid_size}x{env.grid_size} grid")
    print(f"Classifier: {env.config.model_name}")
    print(f"Disaster classes: {list(env.class_to_idx.keys())}")
    print(f"UAV actions: 8-directional movement")
    print("-" * 60)
    
    state = env.reset()
    path_history = [env.uav_position.copy()]
    hazard_log = []
    total_reward = 0
    step_count = 0
    
    print(f"Starting position: {env.uav_position}")
    print(f"Goal position: {env.goal_position}")
    print(f"Hazard zones detected: {np.sum(env.hazard_map > 0.5)} high-risk areas")
    print("\nBeginning navigation...")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.ion()
    
    while step_count < env.max_steps:
        action = agent.select_action(state, deterministic=True)
        
        action_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        print(f"Step {step_count + 1}: Moving {action_names[action]} to {env.uav_position + np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]][action])}")
        
        state, reward, done, info = env.step(action)
        
        path_history.append(info['uav_position'].copy())
        total_reward += reward
        step_count += 1
        
        if info['in_danger_zone']:
            hazard_info = {
                'step': step_count,
                'position': info['uav_position'].copy(),
                'risk_level': info['hazard_level']
            }
            hazard_log.append(hazard_info)
            print(f"  WARNING: Entering hazard zone - Risk level: {info['hazard_level']:.2f}")
        
        ax.clear()
        
        hazard_display = ax.imshow(env.hazard_map, cmap='Reds', alpha=0.8, origin='upper')
        
        if len(path_history) > 1:
            path_array = np.array(path_history)
            ax.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=3, alpha=0.9, label='UAV Path')
        
        current_pos = info['uav_position']
        ax.plot(current_pos[1], current_pos[0], 'ko', markersize=15, label='UAV')
        ax.plot(env.goal_position[1], env.goal_position[0], 'g*', markersize=20, label='Goal')
        
        if len(path_history) > 1:
            start_pos = path_history[0]
            ax.plot(start_pos[1], start_pos[0], 'go', markersize=12, label='Start')
        
        for hazard in hazard_log:
            pos = hazard['position']
            ax.plot(pos[1], pos[0], 'rx', markersize=8, alpha=0.7)
        
        distance_to_goal = np.linalg.norm(current_pos - env.goal_position)
        
        ax.set_xlim(-0.5, env.grid_size-0.5)
        ax.set_ylim(-0.5, env.grid_size-0.5)
        ax.set_title(f'UAV Navigation - Step {step_count} | Distance to Goal: {distance_to_goal:.1f} | Reward: {total_reward:.1f}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.pause(0.5)
        
        if done:
            break
    
    plt.ioff()
    
    mission_success = np.array_equal(info['uav_position'], info['goal_position'])
    final_distance = np.linalg.norm(info['uav_position'] - info['goal_position'])
    
    print("\n" + "=" * 60)
    print("NAVIGATION MISSION COMPLETE")
    print("=" * 60)
    print(f"Mission Status: {'SUCCESS' if mission_success else 'INCOMPLETE'}")
    print(f"Steps taken: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final distance to goal: {final_distance:.2f}")
    print(f"Hazard encounters: {len(hazard_log)}")
    print(f"Path efficiency: {len(path_history)/step_count:.2f}")
    
    if hazard_log:
        print(f"\nHazard Encounter Summary:")
        for i, hazard in enumerate(hazard_log):
            print(f"  {i+1}. Step {hazard['step']}: Risk {hazard['risk_level']:.2f} at {hazard['position']}")
    
    plt.colorbar(hazard_display, ax=ax, label='Disaster Risk Level')
    plt.title(f'Final Navigation Result - {"SUCCESS" if mission_success else "INCOMPLETE"}')
    plt.tight_layout()
    plt.savefig('navigation_demonstration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'success': mission_success,
        'steps': step_count,
        'reward': total_reward,
        'hazard_encounters': len(hazard_log),
        'path_history': path_history
    }

if __name__ == "__main__":
    result = demonstrate_real_time_navigation()
    print(f"\nDemonstration completed with result: {result}")