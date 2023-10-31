import gymnasium as gym
import numpy as np
import random
import torch
from PIL import Image

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model import DQN
from dqn_agent import DQNAgent

from preprocessing import preprocess

def env_reset(env: gym.Env, m):
    reward_total = 0
    states = []
    state = env.reset()[0]
    states.append(state)
    action = 0
    for _ in range(m - 1):
        next_state, reward, done, _, _ = env.step(action)

        reward_total += reward

        states.append(next_state)

        if (done):
            break

    return preprocess(states, m=m), reward_total, done

def env_step(env: gym.Env, action, m):
    states = []
    reward_total = 0

    for _ in range(m):
        next_state, reward, done, _, _ = env.step(action)

        reward_total += reward

        states.append(next_state)

        if done:
            break
    
    return preprocess(states, m=m), reward_total, done

import matplotlib.pyplot as plt


def visualize_preprocessing(stack):
    # Create a grid to display the frames
    rows, cols = 2, 2  # 2x2 grid for the 4 frames
    fig, axs = plt.subplots(rows, cols, figsize=(6, 6))

    for i in range(rows):
        for j in range(cols):
            frame_index = i * cols + j
            frame = stack[frame_index]
            axs[i, j].imshow(frame, cmap='gray')  # Display the frame in grayscale
            axs[i, j].axis('off')  # Turn off axis labels

    plt.tight_layout()
    plt.show()

def play_and_train(env: gym.Env, agent: DQNAgent, m, num_episodes=10000, checkpoint_episode=100):

    checkpoint_reward = 0
    episode_reward = 0
    frame_i = 0
    episode_reward = None
    for episode in range(num_episodes):
        if episode_reward:
            writer.add_scalar('Reward/train', episode_reward, frame_i)
        else:
            writer.add_scalar('Reward/train', 0, frame_i) 

        state, episode_reward, done = env_reset(env, m)

        if done:
            continue
        
        i = 0
        for _ in range(int(1e4)):
            # env.render()
            state_tensor = state[np.newaxis, :]

            action, qvalue_max = agent.get_action(state_tensor, online=True)

            writer.add_scalar('Qvalue/train', qvalue_max, frame_i)
            frame_i += 1

            next_state, reward, done = env_step(env, action, m) 

            episode_reward += reward
            
            i += 1
            if done:
                # state, _, done = env_reset(env, m)
                # continue
                break

            if (agent.add_to_replay_buffer((state, action, reward, next_state))):
                loss = agent.update_online_net()
                writer.add_scalar('Loss/train', loss, episode)

            if agent.if_target_to_update(episode):
                agent.update_target_net()

            
            state = next_state

        checkpoint_reward += episode_reward 
        if episode % checkpoint_episode == 0:
            print(f"Episode {episode}: {checkpoint_reward / checkpoint_episode}")
            checkpoint_reward = 0

    env.close()

log_dir = f"logs_boxing/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

writer = SummaryWriter(log_dir)

# Environment (Only using Breakout for now)
env = gym.make("ALE/Breakout-v5", obs_type="rgb")#, render_mode="human")
agent = DQNAgent(learning_rate=0.01, gamma=0.99, num_actions=env.action_space.n,
                 epsilon_start=1, epsilon_end=0.1, epsilon_decay=150000, memory=100, replay_start_size=5000, target_update_freq=4)

play_and_train(env, agent, 4, checkpoint_episode=1)