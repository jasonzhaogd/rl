import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTIL = 70

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions) -> None:
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )
        
    def forward(self, x):
        return self.net(x)
    
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    obs_size = env.observation_space[0]
    n_actions = env.action_space.n
    
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    write = SummaryWriter(comment="-CartPole")
    
    for i, batch in enumerate(iterate_ba)