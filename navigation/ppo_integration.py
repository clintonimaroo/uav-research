import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from gym_environment import DroneNavigationEnv
from typing import List, Tuple

class PPOPolicy(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], action_dim: int):
        super(PPOPolicy, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        conv_output_size = 64 * 8 * 8
        
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.actor_head = nn.Linear(128, action_dim)
        self.critic_head = nn.Linear(128, 1)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        
        action_logits = self.actor_head(x)
        value = self.critic_head(x)
        
        return action_logits, value

class PPOAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99, eps_clip=0.2, 
                 k_epochs=4, entropy_coef=0.01):
        
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        
        self.policy = PPOPolicy(env.observation_space.shape, env.action_space.n)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.buffer = []
        
    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_logits, value = self.policy(state_tensor)
            
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action).item(), value.item()
    
    def store_transition(self, state, action, reward, next_state, done, 
                        log_prob, value):
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        })
    
    def update(self):
        if len(self.buffer) == 0:
            return
            
        states = torch.FloatTensor([t['state'] for t in self.buffer])
        actions = torch.LongTensor([t['action'] for t in self.buffer])
        rewards = [t['reward'] for t in self.buffer]
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in self.buffer])
        values = torch.FloatTensor([t['value'] for t in self.buffer])
        
        returns = self._compute_returns(rewards)
        returns = torch.FloatTensor(returns)
        
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.k_epochs):
            action_logits, new_values = self.policy(states)
            
            action_probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            entropy_loss = -entropy.mean()
            
            total_loss = actor_loss + 0.5 * critic_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        self.buffer.clear()
    
    def _compute_returns(self, rewards):
        returns = []
        discounted_sum = 0
        
        for reward in reversed(rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
            
        return returns

class DisasterAwarePPO(PPOAgent):
    def __init__(self, env, disaster_classifier=None, **kwargs):
        super().__init__(env, **kwargs)
        self.disaster_classifier = disaster_classifier
        self.disaster_penalty = -100.0
        
    def get_disaster_aware_reward(self, state, base_reward):
        if self.disaster_classifier is None:
            return base_reward
            
        disaster_detected = self._detect_disaster(state)
        
        if disaster_detected:
            return base_reward + self.disaster_penalty
        
        return base_reward
    
    def _detect_disaster(self, state):
        if self.disaster_classifier is None:
            return False
            
        drone_pos = self._get_drone_position(state)
        local_region = self._extract_local_region(state, drone_pos)
        
        return self.disaster_classifier.predict_disaster(local_region)
    
    def _get_drone_position(self, state):
        drone_channel = state[:, :, 0]
        pos = np.where(drone_channel == 1.0)
        return np.array([pos[0][0], pos[1][0]])
    
    def _extract_local_region(self, state, center_pos, region_size=5):
        h, w = state.shape[:2]
        x, y = center_pos
        
        x_start = max(0, x - region_size // 2)
        x_end = min(h, x + region_size // 2 + 1)
        y_start = max(0, y - region_size // 2)
        y_end = min(w, y + region_size // 2 + 1)
        
        return state[x_start:x_end, y_start:y_end]

def train_ppo_agent(episodes=1000):
    env = DroneNavigationEnv(grid_size=15, max_steps=200)
    agent = PPOAgent(env)
    
    scores = []
    update_frequency = 32
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action, log_prob, value = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done, 
                                 log_prob, value)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        scores.append(episode_reward)
        
        if len(agent.buffer) >= update_frequency:
            agent.update()
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
    
    return agent

if __name__ == "__main__":
    print("Training PPO Agent")
    trained_agent = train_ppo_agent(episodes=500)