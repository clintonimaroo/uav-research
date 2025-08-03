import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Tuple

from uav_environment import UAVNavigationEnv
from ppo_navigation import PPONavigationAgent

class UAVNavigationSystem:
    def __init__(self, grid_size: int = 50, max_episode_steps: int = 200):
        self.env = UAVNavigationEnv(grid_size=grid_size, max_steps=max_episode_steps)
        self.agent = PPONavigationAgent(
            input_shape=self.env.observation_space.shape,
            action_dim=self.env.action_space.n,
            learning_rate=1e-4,
            gamma=0.99
        )
        
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_episodes': [],
            'policy_losses': [],
            'navigation_efficiency': []
        }
        
        self.checkpoint_dir = "navigation_models"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def train_navigation_system(self, total_episodes: int = 2000, 
                               update_frequency: int = 32, 
                               save_frequency: int = 200,
                               log_frequency: int = 100):
        
        print(f"Training UAV Navigation System")
        print(f"Environment: {self.env.grid_size}x{self.env.grid_size} grid")
        print(f"Classifier: {self.env.config.model_name}")
        print(f"Target Episodes: {total_episodes}")
        print("-" * 60)
        
        episode_count = 0
        best_success_rate = 0.0
        
        while episode_count < total_episodes:
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            
            while True:
                action = self.agent.select_action(state, deterministic=False)
                next_state, reward, done, info = self.env.step(action)
                
                self.agent.experience_buffer.rewards.append(reward)
                self.agent.experience_buffer.terminals.append(done)
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
            
            self.training_metrics['episode_rewards'].append(episode_reward)
            self.training_metrics['episode_lengths'].append(episode_steps)
            
            success = episode_reward > 100
            self.training_metrics['success_episodes'].append(success)
            
            goal_distance = np.linalg.norm(info['uav_position'] - info['goal_position'])
            efficiency = 1.0 / (1.0 + goal_distance + episode_steps * 0.01)
            self.training_metrics['navigation_efficiency'].append(efficiency)
            
            if (episode_count + 1) % update_frequency == 0:
                loss_info = self.agent.update_policy()
                self.training_metrics['policy_losses'].append(loss_info['total_loss'])
            
            if (episode_count + 1) % log_frequency == 0:
                recent_rewards = self.training_metrics['episode_rewards'][-log_frequency:]
                recent_successes = self.training_metrics['success_episodes'][-log_frequency:]
                recent_efficiency = self.training_metrics['navigation_efficiency'][-log_frequency:]
                
                avg_reward = np.mean(recent_rewards)
                success_rate = np.mean(recent_successes)
                avg_efficiency = np.mean(recent_efficiency)
                
                print(f"Episode {episode_count + 1}/{total_episodes}")
                print(f"Avg Reward: {avg_reward:.2f} | Success Rate: {success_rate:.2%} | Efficiency: {avg_efficiency:.3f}")
                
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    self.save_best_model()
            
            if (episode_count + 1) % save_frequency == 0:
                self.save_checkpoint(episode_count + 1)
                self.plot_training_metrics()
                
                if success:
                    self.demonstrate_navigation(episode_count + 1, save_visualization=True)
            
            episode_count += 1
        
        print(f"\nTraining completed. Best success rate: {best_success_rate:.2%}")
        self.generate_final_report()
    
    def demonstrate_navigation(self, episode_id: int = None, save_visualization: bool = False,
                              real_time: bool = True):
        
        self.load_best_model()
        
        state = self.env.reset()
        path_positions = [self.env.uav_position.copy()]
        rewards_log = []
        hazard_encounters = []
        
        step_count = 0
        total_reward = 0
        
        if real_time:
            fig, ax = plt.subplots(figsize=(12, 10))
            
        while step_count < self.env.max_steps:
            action = self.agent.select_action(state, deterministic=True)
            state, reward, done, info = self.env.step(action)
            
            path_positions.append(info['uav_position'].copy())
            rewards_log.append(reward)
            total_reward += reward
            
            if info['in_danger_zone']:
                hazard_encounters.append({
                    'position': info['uav_position'].copy(),
                    'hazard_level': info['hazard_level'],
                    'step': step_count
                })
            
            if real_time:
                self._update_real_time_display(ax, path_positions, hazard_encounters)
                plt.pause(0.1)
            
            step_count += 1
            
            if done:
                break
        
        navigation_result = {
            'success': np.array_equal(info['uav_position'], info['goal_position']),
            'total_reward': total_reward,
            'steps_taken': step_count,
            'path_length': len(path_positions),
            'hazard_encounters': len(hazard_encounters),
            'final_distance': np.linalg.norm(info['uav_position'] - info['goal_position'])
        }
        
        if save_visualization:
            self._save_navigation_visualization(path_positions, hazard_encounters, 
                                              navigation_result, episode_id)
        
        self._print_navigation_summary(navigation_result, hazard_encounters)
        
        return navigation_result
    
    def _update_real_time_display(self, ax, path_positions, hazard_encounters):
        ax.clear()
        
        hazard_display = ax.imshow(self.env.hazard_map, cmap='Reds', alpha=0.7, origin='upper')
        
        if len(path_positions) > 1:
            path_array = np.array(path_positions)
            ax.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=2, alpha=0.8)
        
        current_pos = path_positions[-1]
        ax.plot(current_pos[1], current_pos[0], 'ko', markersize=15)
        ax.plot(self.env.goal_position[1], self.env.goal_position[0], 'g*', markersize=20)
        
        for encounter in hazard_encounters:
            pos = encounter['position']
            ax.plot(pos[1], pos[0], 'rx', markersize=8, alpha=0.7)
        
        ax.set_xlim(-0.5, self.env.grid_size-0.5)
        ax.set_ylim(-0.5, self.env.grid_size-0.5)
        ax.set_title(f'UAV Real-Time Navigation - Step {len(path_positions)}')
        ax.grid(True, alpha=0.3)
    
    def _save_navigation_visualization(self, path_positions, hazard_encounters, 
                                     result, episode_id):
        fig, ax = plt.subplots(figsize=(14, 12))
        
        hazard_display = ax.imshow(self.env.hazard_map, cmap='Reds', alpha=0.8, origin='upper')
        
        path_array = np.array(path_positions)
        ax.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=3, label='UAV Path')
        
        ax.plot(path_positions[0][1], path_positions[0][0], 'go', markersize=15, label='Start')
        ax.plot(path_positions[-1][1], path_positions[-1][0], 'ko', markersize=15, label='Final Position')
        ax.plot(self.env.goal_position[1], self.env.goal_position[0], 'g*', markersize=25, label='Goal')
        
        for encounter in hazard_encounters:
            pos = encounter['position']
            ax.plot(pos[1], pos[0], 'rx', markersize=10, alpha=0.8)
        
        status = "SUCCESS" if result['success'] else "INCOMPLETE"
        title = f'UAV Navigation {status} - Reward: {result["total_reward"]:.1f}'
        
        ax.set_xlim(-0.5, self.env.grid_size-0.5)
        ax.set_ylim(-0.5, self.env.grid_size-0.5)
        ax.set_title(title, fontsize=16)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(hazard_display, ax=ax, label='Disaster Risk Level')
        
        output_path = f"navigation_results/episode_{episode_id}_navigation.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_navigation_summary(self, result, hazard_encounters):
        print("\nNavigation Mission Summary:")
        print("-" * 40)
        print(f"Mission Status: {'SUCCESS' if result['success'] else 'INCOMPLETE'}")
        print(f"Total Reward: {result['total_reward']:.2f}")
        print(f"Steps Taken: {result['steps_taken']}")
        print(f"Path Efficiency: {result['path_length']/result['steps_taken']:.2f}")
        print(f"Hazard Encounters: {result['hazard_encounters']}")
        print(f"Final Distance to Goal: {result['final_distance']:.2f}")
        
        if hazard_encounters:
            print("\nHazard Encounter Details:")
            for i, encounter in enumerate(hazard_encounters):
                print(f"  {i+1}. Step {encounter['step']}: Risk Level {encounter['hazard_level']:.2f}")
    
    def save_checkpoint(self, episode: int):
        model_path = f"{self.checkpoint_dir}/uav_navigation_{episode}.pth"
        self.agent.save_model(model_path)
        
        metrics_path = f"{self.checkpoint_dir}/training_metrics_{episode}.json"
        
        serializable_metrics = {}
        for key, values in self.training_metrics.items():
            if isinstance(values, list):
                serializable_metrics[key] = [float(v) if hasattr(v, 'dtype') else v for v in values]
            else:
                serializable_metrics[key] = values
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
    
    def save_best_model(self):
        best_model_path = f"{self.checkpoint_dir}/best_navigation_model.pth"
        self.agent.save_model(best_model_path)
    
    def load_best_model(self):
        best_model_path = f"{self.checkpoint_dir}/best_navigation_model.pth"
        return self.agent.load_model(best_model_path)
    
    def plot_training_metrics(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(1, len(self.training_metrics['episode_rewards']) + 1)
        
        axes[0, 0].plot(episodes, self.training_metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        if len(self.training_metrics['success_episodes']) >= 100:
            success_rate = []
            for i in range(99, len(self.training_metrics['success_episodes'])):
                rate = np.mean(self.training_metrics['success_episodes'][i-99:i+1])
                success_rate.append(rate)
            
            axes[0, 1].plot(range(100, len(episodes) + 1), success_rate)
            axes[0, 1].set_title('Success Rate (100-episode window)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Success Rate')
            axes[0, 1].grid(True)
        
        axes[1, 0].plot(episodes, self.training_metrics['episode_lengths'])
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(episodes, self.training_metrics['navigation_efficiency'])
        axes[1, 1].set_title('Navigation Efficiency')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Efficiency Score')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('navigation_training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_final_report(self):
        total_episodes = len(self.training_metrics['episode_rewards'])
        final_success_rate = np.mean(self.training_metrics['success_episodes'][-100:]) if total_episodes >= 100 else np.mean(self.training_metrics['success_episodes'])
        avg_reward = np.mean(self.training_metrics['episode_rewards'][-100:]) if total_episodes >= 100 else np.mean(self.training_metrics['episode_rewards'])
        avg_efficiency = np.mean(self.training_metrics['navigation_efficiency'][-100:]) if total_episodes >= 100 else np.mean(self.training_metrics['navigation_efficiency'])
        
        report = {
            'training_completed': datetime.now().isoformat(),
            'total_episodes': total_episodes,
            'final_success_rate': float(final_success_rate),
            'average_reward': float(avg_reward),
            'navigation_efficiency': float(avg_efficiency),
            'classifier_model': self.env.config.model_name,
            'environment_size': f"{self.env.grid_size}x{self.env.grid_size}",
            'disaster_classes': list(self.env.class_to_idx.keys())
        }
        
        with open('navigation_training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nFinal Training Report:")
        print(f"Success Rate: {final_success_rate:.2%}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Navigation Efficiency: {avg_efficiency:.3f}")

def main():
    navigation_system = UAVNavigationSystem(grid_size=50, max_episode_steps=200)
    
    print("Initializing UAV Navigation System with Disaster Detection")
    print("Training autonomous navigation agent...")
    
    navigation_system.train_navigation_system(
        total_episodes=2000,
        update_frequency=32,
        save_frequency=200,
        log_frequency=50
    )
    
    print("\nDemonstrating trained navigation system...")
    navigation_system.demonstrate_navigation(real_time=True, save_visualization=True)

if __name__ == "__main__":
    main()