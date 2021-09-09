import gym
import carla_env
from ResizeObservation import ObservationWrapper


import d3rlpy
from d3rlpy.dataset import MDPDataset

from wrappers.collector import DataCollector

import imageio
import numpy as np


env = gym.make('CarlaEnv-pixel-v1')
env = ObservationWrapper(env)


env.reset()
for i in range(50000):
    next_obs, reward, done, info = env.step([1, 0])
 
env.close()

