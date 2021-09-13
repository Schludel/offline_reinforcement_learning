import d3rlpy
from d3rlpy.dataset import MDPDataset

import torch

import numpy as np
from xvfbwrapper import Xvfb

import gym
import carla_env

from ResizeObservation import ObservationWrapper
from frame_stack_wrapper import FrameStack

# prepare dataset
env = gym.make('CarlaEnv-pixel-v1')
env = ObservationWrapper(env)
env = FrameStack(env, 4)

# prepare algorithm
cql = d3rlpy.algos.CQL(use_gpu=True)

#cql.build_with_dataset(dataset)
cql.build_with_env(env)

cql.load_model("./d3rlpy/Carla/d3rlpy_logs/expert_dataset_50k_new_reward_noise_0.6_alpha_0.1_20210811105023/model_804.pt")



obs = env.reset()
done = False
while not done:
    action = cql.predict([obs.copy()])[0]
    obs, reward, done, info = env.step(action)

env.close

