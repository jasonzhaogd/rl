import gym

env = gym.make("FrozenLake-v1")

print(env.observation_space.n)

obs = env.reset()

x = env.step(1)

print(x)