import random
import torch

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

        self.replay_buffer = []

        self.target_update_freq = target_update_freq

        self.online_net = DQN(self.num_actions)
        self.target_net = DQN(self.num_actions)

        self.target_net.load_state_dict(self.online_net.state_dict())

        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.learning_rate) 

    def loss_function(self, qvalues_target, qvalues_online):
        return torch.mean((qvalues_target - qvalues_online) ** 2)
    
    def get_qvalues(self, state: torch.Tensor, online: bool):
        return self.online_net(state) if online else self.target_net(state)
    
    def get_action(self, state: torch.Tensor, online: bool):
        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions - 1)
        else:
            qvalues = self.get_qvalues(state, online)
            action = qvalues.argmax().item()
            
        self.epsilon = max(self.epsilon_end, self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.timestep / self.epsilon_decay)

        return action
    
    def add_to_replay_buffer(self, transition):
        self.replay_buffer.append(transition) # transition = (state, action, reward, next_state, done)
        return len(self.replay_buffer) >= self.batch_size

    def update_online_net(self):
        batch = random.sample(self.replay_buffer, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        self.online_net.zero_grad()

        qvalues_online = self.online_net(states).gather(1, actions.unsqueeze(1))
        
        qvalues_next = self.target_net(next_states).max(1)[0].detach()

        qvalues_target = rewards + self.gamma * qvalues_next

        loss = self.loss_function(qvalues_target.unsqueeze(1), qvalues_online)
        loss.backward()
        self.optimizer.step()

    def if_target_to_update(self, episode: int):
        return episode % self.target_update_freq == 0

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
