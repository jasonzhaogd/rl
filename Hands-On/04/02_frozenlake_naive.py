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

class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        shape = (env.observation_space.n, )
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape, dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res

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

def iterate_batch(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    softmax = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = softmax(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if batch_size == len(batch):
                yield batch
                batch = []
        obs = next_obs

def filter_batch(batch, percentile):
    rewards = list(map(lambda s:s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))
    
    train_obs = []
    train_act = []
    
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda s:s.observation, steps))
        train_act.extend(map(lambda s:s.action, steps))
    
    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

if __name__ == "__main__":
    env = DiscreteOneHotWrapper(gym.make("FrozenLake-v1"))
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    write = SummaryWriter(comment="-FrozenLake-Naive")
    
    for i, batch in enumerate(iterate_batch(env, net, BATCH_SIZE)):
        obs_v, act_v, reward_b, reward_m = filter_batch(batch, PERCENTIL)
        optimizer.zero_grad()
        action_score_v = net(obs_v)
        loss = objective(action_score_v, act_v)
        loss.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (i, loss.item(), reward_m, reward_b))
        write.add_scalar("loss", loss.item(), i)
        write.add_scalar("reward_bound", reward_b, i)
        write.add_scalar("reward_mean", reward_m, i)
        if reward_m > 0.8:
            print("Solved!")
            break
        
    write.close()
