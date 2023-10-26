import gymnasium as gym
from model import DQN
import numpy as np
from preprocessing import preprocess
import torch

env = gym.make("ALE/Breakout-v5", obs_type="rgb")

action = 3
env.reset()

dqnModel = DQN(4)

frames = []
for i in range(4):
    next_state, reward, done, _, _ = env.step(action)
    frames.append(next_state)

input = preprocess(frames, 4)


print(torch.tensor(input, dtype=torch.float32).unsqueeze(0).shape)

qvalues = dqnModel(torch.tensor(input, dtype=torch.float32).unsqueeze(0))


print(qvalues)
print(qvalues.argmax().item())