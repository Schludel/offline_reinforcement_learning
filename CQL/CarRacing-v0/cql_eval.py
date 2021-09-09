import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.preprocessing.stack import StackedObservation

import torch

import numpy as np
import pickle
import os
from xvfbwrapper import Xvfb

import gym

from ResizeObservation import ObservationWrapper
from frame_stack_wrapper import FrameStack

# prepare dataset
env = gym.make('CarRacing-v0')
env = ObservationWrapper(env)
env = FrameStack(env, 4)


# prepare algorithm
cql = d3rlpy.algos.CQL(use_gpu=True)

#cql.build_with_dataset(dataset)
cql.build_with_env(env)

cql.load_model("/home/ws/ujvhi/d3rlpy/CarRacing-v0/d3rlpy_logs/expert_dataset_merged_learn_alpha_0.01_threshold_10.0_20210720120854/model_57.pt")

#stacked_obs = StackedObservation(env.observation_space.shape, 1)


obs = env.reset()
done = False
while not done:
    #next_obs, reward, done, info = env.step([1, 0])
    action = cql.predict([obs.copy()])[0]
    print('action', action)
    obs, reward, done, info = env.step(action)
    env.render()

env.close