import random
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque 

from model import DQN

class DQNAgent:

    def __init__(
            self,
            learning_rate: float,
            gamma: float,
            num_actions: int,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.1,
            epsilon_decay: float = 1e6,
            batch_size: float = 32,
            memory: float = 1e6,
            replay_start_size = 5000,
            target_update_freq: int = 100
    ):
        '''
        DQN Agent

        '''
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.num_actions = num_actions

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.timestep = 0

        self.batch_size = batch_size
        self.memory = memory
        self.replay_buffer = deque()
        self.replay_start_size = replay_start_size

        # self.replay_buffer = []

        self.target_update_freq = target_update_freq

        self.online_net = DQN(self.num_actions)
        self.target_net = DQN(self.num_actions)

        self.target_net.load_state_dict(self.online_net.state_dict())

        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate, eps=1e-6)
 

    def loss_function(self, qvalues_target, qvalues_online):
        return F.huber_loss(qvalues_online, qvalues_target)
        # return torch.mean((qvalues_target - qvalues_online) ** 2)
    
    def get_qvalues(self, state: torch.Tensor, online: bool):
        return self.online_net(state) if online else self.target_net(state)
    
    def get_action(self, state: torch.Tensor, online: bool):
        if random.random() < self.epsilon or self.replay_start_size != 0:
            action = random.randint(0, self.num_actions - 1)
        else:
            qvalues = self.get_qvalues(torch.tensor(state, dtype=torch.float32), online)
            action = qvalues.argmax().item()

        if self.replay_start_size == 0:    
            self.epsilon = max(self.epsilon_end, self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.timestep / self.epsilon_decay)
            self.timestep += 1
        
        qvalues = self.get_qvalues(torch.tensor(state, dtype=torch.float32), online)
        qvalue_max = torch.max(qvalues).item()

        return action, qvalue_max, self.epsilon
    
    def add_to_replay_buffer(self, transition):
        if (len(self.replay_buffer) == self.memory):
            self.replay_buffer.pop()
        self.replay_buffer.append(transition) # transition = (state, action, reward, next_state, done)

        if self.replay_start_size > 0:
            self.replay_start_size -= 1

        return len(self.replay_buffer) == self.memory and self.replay_start_size == 0

    def update_online_net(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        self.online_net.zero_grad()

        qvalues_online = self.online_net(states).gather(1, actions.unsqueeze(1))

        qvalues_next = self.target_net(next_states).max(1)[0].detach()

        qvalues_target = rewards + self.gamma * qvalues_next

        loss = self.loss_function(qvalues_target.unsqueeze(1), qvalues_online)
        # print(loss)
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def if_target_to_update(self, episode: int):
        return episode % self.target_update_freq == 0

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
