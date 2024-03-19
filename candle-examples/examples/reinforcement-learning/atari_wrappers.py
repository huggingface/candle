import gymnasium as gym
import numpy as np

def make_env(env_name, video_folder):
    if video_folder is not None:
        env = gym.make(env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_folder)
    else:
        env = gym.make(env_name)
    env = gym.wrappers.AtariPreprocessing(env, scale_obs=True)
    env = gym.wrappers.TransformReward(env, lambda reward: np.sign(reward))
    return env

def make(env_name, num_processes, video_folder):
    envs = gym.vector.AsyncVectorEnv([lambda: make_env(env_name, video_folder) for _ in range(num_processes)]) 
    return envs
