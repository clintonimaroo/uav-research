import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from typing import List, Tuple, Dict
import os

class NavigationNetwork(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], action_dim: int, hidden_dim: int = 512):
        super(NavigationNetwork, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[2], 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        feature_size = 128 * 8 * 8
        
        self.shared_network = nn.Sequential(
            nn.Linear(feature_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        
        self.policy_head = nn.Linear(hidden_dim//2, action_dim)
        self.value_head = nn.Linear(hidden_dim//2, 1)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        x = x.permute(0, 3, 1, 2)
        features = self.feature_extractor(x)
        features = features.contiguous().view(features.size(0), -1)
        shared_output = self.shared_network(features)
        
        policy_logits = self.policy_head(shared_output)
        value = self.value_head(shared_output)
        
        return policy_logits, value

class PPONavigationAgent:
    def __init__(self, input_shape: Tuple[int, int, int], action_dim: int,
                 learning_rate: float = 2e-4, gamma: float = 0.99,
                 epsilon_clip: float = 0.2, update_epochs: int = 10):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.update_epochs = update_epochs
        
        self.policy_net = NavigationNetwork(input_shape, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        self.old_policy_net = NavigationNetwork(input_shape, action_dim).to(self.device)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        
        self.experience_buffer = ExperienceBuffer()
        
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.old_policy_net(state_tensor)
            action_probs = F.softmax(policy_logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
                log_prob = torch.log(action_probs[0, action])
            else:
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
        
        self.experience_buffer.states.append(state)
        self.experience_buffer.actions.append(action.item())
        self.experience_buffer.log_probs.append(log_prob.item())
        self.experience_buffer.values.append(value.item())
        
        return action.item()
    
    def update_policy(self):
        rewards = self._compute_discounted_rewards()
        
        states = torch.stack([torch.FloatTensor(s) for s in self.experience_buffer.states]).to(self.device)
        actions = torch.tensor(self.experience_buffer.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(self.experience_buffer.log_probs).to(self.device)
        old_values = torch.tensor(self.experience_buffer.values).to(self.device)
        
        advantages = rewards - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_losses = []
        value_losses = []
        
        for _ in range(self.update_epochs):
            policy_logits, values = self.policy_net(states)
            action_probs = F.softmax(policy_logits, dim=-1)
            dist = Categorical(action_probs)
            
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values.squeeze(), rewards)
            
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
        
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        self.experience_buffer.clear()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'total_loss': np.mean(policy_losses) + 0.5 * np.mean(value_losses)
        }
    
    def _compute_discounted_rewards(self):
        rewards = []
        discounted_reward = 0
        
        for reward, terminal in zip(reversed(self.experience_buffer.rewards), 
                                   reversed(self.experience_buffer.terminals)):
            if terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        return (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            return False
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.old_policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return True

class ExperienceBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.terminals = []
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.terminals.clear()