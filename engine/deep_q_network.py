import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DeepQNetwork:
    def __init__(self, state_size, action_size, model_name=None, load_existing_model=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model_name = model_name
        if load_existing_model:
            self.model = self.load_model()
        else:
            self.model = self._build_model()
        self.q_models = {}
        self.last_action = None
        self.last_state = None

    def load_model(self):
        """Load model from file if exists."""
        model_path = f"saved_models/{self.model_name}.pt"
        if os.path.exists(model_path):
            return torch.load(model_path)
        else:
            print(f"No model found at {model_path}. Building a new model.")
            return self._build_model()

    def _build_model(self):
        """Neural Network for Deep Q learning Model."""
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
            next_state = torch.from_numpy(np.array(next_state)).float().unsqueeze(0)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

            if not done:
                target = (reward + self.gamma * torch.max(self.model(next_state)).item())
            else:
                target = reward

            target = target.float()

            action = action.long()  # Ensure the action is a Long tensor
            current_q = self.model(state)[0, action]
            loss = nn.MSELoss()(current_q, target)

            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, model_name, tick):
        torch.save(self.model, f"saved_models/{model_name}-{tick}.pt")
