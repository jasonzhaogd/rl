import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# load evn
env = gym.make('CartPole-v0')

# print(env.observation_space)

# test out the env
# episodes = 5
# for episode in range(1, episodes + 1):
#     state = env.reset()
#     done = False
#     score = 0
    
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward
        
#     print('Episode:{}, Score:{}'.format(episode, score))
    
# env.close()


# training model
log_path = os.path.join('training', 'logs')
env = DummyVecEnv([lambda: env]) # type: ignore
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=20000)
