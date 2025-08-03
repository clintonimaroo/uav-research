import numpy as np
from gym_environment import DroneNavigationEnv
from astar_baseline import BaselineAgent
from ppo_integration import PPOAgent, DisasterAwarePPO
from disaster_classifier_bridge import create_disaster_detector

def compare_agents():
    print("Comparative Analysis: A* Baseline vs PPO vs Disaster-Aware PPO")
    print("=" * 70)
    
    env = DroneNavigationEnv(grid_size=15, max_steps=150)
    
    astar_agent = BaselineAgent(grid_size=15)
    ppo_agent = PPOAgent(env)
    
    model_path = "../checkpoints/best_model.pth"
    config_path = "../checkpoints/config.yaml"
    disaster_detector = create_disaster_detector(model_path, config_path, use_real_model=False)
    disaster_ppo = DisasterAwarePPO(env, disaster_classifier=disaster_detector)
    
    agents = {
        "A* Baseline": astar_agent,
        "PPO": ppo_agent,
        "Disaster-Aware PPO": disaster_ppo
    }
    
    num_episodes = 20
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"\nTesting {agent_name}:")
        print("-" * 40)
        
        successes = 0
        total_steps = 0
        total_rewards = 0
        
        for episode in range(num_episodes):
            obs = env.reset()
            if hasattr(agent, 'reset'):
                agent.reset()
            
            episode_steps = 0
            episode_reward = 0
            
            while True:
                if agent_name == "A* Baseline":
                    action = agent.get_action(obs)
                else:
                    action, _, _ = agent.get_action(obs)
                
                obs, reward, done, _ = env.step(action)
                
                if agent_name == "Disaster-Aware PPO":
                    reward = agent.get_disaster_aware_reward(obs, reward)
                
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            if episode_reward > 50:
                successes += 1
            
            total_steps += episode_steps
            total_rewards += episode_reward
        
        success_rate = (successes / num_episodes) * 100
        avg_steps = total_steps / num_episodes
        avg_reward = total_rewards / num_episodes
        
        results[agent_name] = {
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_reward': avg_reward
        }
        
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Average Reward: {avg_reward:.1f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    
    for agent_name, metrics in results.items():
        print(f"{agent_name:20} | Success: {metrics['success_rate']:5.1f}% | "
              f"Steps: {metrics['avg_steps']:5.1f} | Reward: {metrics['avg_reward']:6.1f}")

def test_disaster_integration():
    print("\nTesting Disaster Detection Integration")
    print("=" * 50)
    
    env = DroneNavigationEnv(grid_size=10, max_steps=50)
    
    model_path = "../checkpoints/best_model.pth"
    config_path = "../checkpoints/config.yaml"
    
    disaster_detector = create_disaster_detector(model_path, config_path, use_real_model=True)
    
    obs = env.reset()
    
    print("Environment state:")
    env.render()
    
    drone_pos = np.where(obs[:, :, 0] == 1.0)
    drone_position = np.array([drone_pos[0][0], drone_pos[1][0]])
    
    local_region = obs[max(0, drone_position[0]-2):drone_position[0]+3,
                         max(0, drone_position[1]-2):drone_position[1]+3]
    
    disaster_detected = disaster_detector.predict_disaster(local_region)
    disaster_type, confidence = disaster_detector.get_disaster_type(local_region)
    
    print(f"\nDisaster Detection Results:")
    print(f"Disaster Detected: {disaster_detected}")
    print(f"Type: {disaster_type}")
    print(f"Confidence: {confidence:.3f}")

if __name__ == "__main__":
    compare_agents()
    test_disaster_integration()