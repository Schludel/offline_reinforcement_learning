import os
import numpy as np 
import h5py

import d3rlpy
from d3rlpy.dataset import MDPDataset


with h5py.File('/home/ws/ujvhi/d3rlpy/Carla/data_collection/expert_dataset_50k_new_reward_noise_0.8.h5', 'r') as f:
    observations = f['observations'][()]
    actions = f['actions'][()]
    rewards = f['rewards'][()]
    terminals = f['terminals'][()]
    print(len(observations))
    print(len(actions))
    print(len(rewards))
    print(len(terminals))
    print(sum(rewards))
    print(500 * sum(rewards)/50000)