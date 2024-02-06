import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import  core
from envs.cartpole import Cartpole
from utils.logx import EpochLogger
from torch.nn.functional import softplus
from filters.NN_filters import FCN
from envs.HalfCheetah import HalfCheetahEnv

def run_episode(env, policy, max_steps=1000):
    obss = []
    obs = env.reset()[0]
    obss.append(obs)
    total_reward = 0

    for step in range(max_steps):
        a, v, vc, logp = policy.step(torch.as_tensor(obs, dtype=torch.float32))
        next_obs, reward, done, _ = env.step(a)
        total_reward += reward
        obs= next_obs
        obss.append(obs)

        if done:
            break

    return total_reward, obss

if __name__ == "__main__":
    p_type = 0
    seed = 3
    if p_type == 0: # unconstrained ppo
        pret_dir = f"data/ppo/ppo-HalfCheetah_s{seed}/pyt_save/model.pt"
        policy = torch.load(pret_dir)
    elif p_type == 1: # RL-DH ppo only valid on seeds 0, 3, 4, 5, 6
        pret_dir = f"data/pret_ppo/reandom_train_5e6/pret_ppo-HalfCheetah_s{seed}/pyt_save/model.pt"
        policy = torch.load(pret_dir)
    elif p_type == 2: # PPO Lag
        pret_dir = f"data/ppo_lag/ppo_lag-HalfCheetah_s{seed}/pyt_save/model.pt"
        policy = torch.load(pret_dir)

    env_fn = lambda : HalfCheetahEnv()
    env = env_fn()

    num_episodes = 1

    for episode in range(num_episodes):
        total_reward, obss = run_episode(env, policy)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()