import gymnasium as gym
import numpy as np
import random
import torch

from model import DQN
from dqn_agent import DQNAgent

from preprocessing import preprocess

def play_and_train(env: gym.Env, agent: DQNAgent):
    num_episodes = 1000

    for episode in range(num_episodes):
        state = env.reset()
        state = state[0]
        state = preprocess([state], m=1)

        episode_reward = 0
        done = False

        while not done:
            # state_tensor = state.unsqueeze(0)
            state_tensor = state[np.newaxis, :]

            action = agent.get_action(state_tensor, online=True)

            next_state, reward, done, _, _ =  env.step(action)
            episode_reward += reward
            next_state = preprocess([next_state], m=1)


            if (agent.add_to_replay_buffer((state, action, reward, next_state, done))):
                agent.update_online_net()

            if agent.if_target_to_update(episode):
                agent.update_target_net()
            
            state = next_state
        
        print(f"Episode {episode}: {episode_reward}")

    env.close()

# Environment (Only using Breakout for now)
env = gym.make("ALE/Breakout-v5", obs_type="rgb")

agent = DQNAgent(learning_rate=0.001, gamma=0.99, num_actions=env.action_space.n)

play_and_train(env, agent)