from preprocessing import preprocess
from PIL import Image
import numpy as np

import gymnasium as gym

env = gym.make("ALE/Breakout-v5", obs_type="rgb")

action = 3
env.reset()
for episode in range(5):
    next_state, reward, done, _, _ = env.step(action)
    frames = preprocess([next_state], m=1)
    frame = Image.fromarray(next_state)
    frame.save(f'frame_color-{episode}.png')
    for i in range(frames.shape[2]):
        frame = Image.fromarray(frames[:, :, i])
        frame.save(f'frame_{i}-{episode}.png')

    print(episode)

