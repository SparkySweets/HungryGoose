from collections import deque
import torch
import pandas as pd

# hungry-geese imports
from kaggle_environments import make
from dqn import DQN
from Coach import q_learning
from replay_memory import ReplayMemory

env = make("hungry_geese", debug=False)

n_episode = 10000
replay_size = 64
target_update = 20
dqn = DQN(12)
dqn.model.load_state_dict(torch.load('convgoose.net'))
memory = ReplayMemory(capacity=100000)

rewards = q_learning(env, dqn, memory, 'non_greedy', n_episode, replay_size, target_update, gamma = 1, epsilon = .15)
torch.save(dqn.model.state_dict(), 'convgoose.net')
reward_table = pd.DataFrame(rewards)
reward_table.to_csv('out.csv')