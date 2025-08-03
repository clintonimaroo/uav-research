import numpy as np
from gym_environment import DroneNavigationEnv
from astar_baseline import BaselineAgent

def test_baseline_navigation():
    env = DroneNavigationEnv(grid_size=15, max_steps=200)
    agent = BaselineAgent(grid_size=15)
    
    num_episodes = 10
    success_count = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        agent.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.get_action(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        success = reward > 50
        if success:
            success_count += 1
            
        print(f"Episode {episode + 1}: Steps={steps}, Reward={total_reward:.1f}, Success={success}")
    
    print(f"\nBaseline A* Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")

def visualize_episode():
    env = DroneNavigationEnv(grid_size=10, max_steps=50)
    agent = BaselineAgent(grid_size=10)
    
    obs = env.reset()
    agent.reset()
    
    print("Initial State:")
    env.render()
    print()
    
    step = 0
    while True:
        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        step += 1
        
        print(f"Step {step}, Action: {action}, Reward: {reward:.1f}")
        env.render()
        print()
        
        if done or step > 20:
            break

if __name__ == "__main__":
    print("Testing A* Baseline Navigation")
    print("=" * 40)
    test_baseline_navigation()
    
    print("\nVisualization Example:")
    print("=" * 40)
    visualize_episode()