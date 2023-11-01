from preprocessing import preprocess
from PIL import Image
import numpy as np

import gymnasium as gym

# env = gym.make("ALE/Breakout-v5", obs_type="rgb")
# 
# action = 3
# state = env.reset()
# frames = preprocess(state, m=1)
# 
# # print(state.shape)
# frame = Image.fromarray(state[0])
# frame.save(f'initial_state.png')
# 
# for episode in range(5):
#     next_state, reward, done, _, _ = env.step(action)
#     frames = preprocess([next_state], m=1, visualize=True)
#     frame = Image.fromarray(next_state)
#     frame.save(f'frame_color-{episode}.png')
#     for i in range(frames.shape[2]):
#         frame = Image.fromarray(frames[:, :, i])
#         frame.save(f'frame_{i}-{episode}.png')

import gymnasium as gym
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers import GrayScaleObservation
from gymnasium.wrappers import ResizeObservation

env = gym.make("ALE/Breakout-v5")
env = GrayScaleObservation(env)
env = ResizeObservation(env, 84)
env = FrameStack(env, 4)

state, _ = env.reset()

state = preprocess(state)

state, reward, done, _, _ = env.step(1)
state = preprocess(state)
for i in range(4):
    frame = Image.fromarray(state[i])
    frame.save(f'frame_wrapper1-{i}.png')

state, reward, done, _, _ = env.step(3)
state = preprocess(state)
for i in range(4):
    frame = Image.fromarray(state[i])
    frame.save(f'frame_wrapper2-{i}.png')