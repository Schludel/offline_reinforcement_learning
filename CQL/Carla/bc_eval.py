import d3rlpy
from d3rlpy.dataset import MDPDataset

import skvideo.io
import numpy as np
import pickle
import os
from xvfbwrapper import Xvfb

import gym
import carla_env

from ResizeObservation import ObservationWrapper
from frame_stack_wrapper import FrameStack

# prepare dataset
env = gym.make('CarlaEnv-pixel-v1')
env = ObservationWrapper(env)
env = FrameStack(env, 4)


dataset = MDPDataset.load('/home/ws/ujvhi/d3rlpy/Carla/data_collection/expert_dataset_100k.h5')

iterator = 500
#video dataset
frame_dataset = np.zeros(shape=(iterator, 84, 84, 3))

# prepare algorithm
bc = d3rlpy.algos.BC(use_gpu=True)

bc.build_with_dataset(dataset)

bc.load_model("/home/ws/ujvhi/d3rlpy/Carla/results_bc/bc_expert_dataset_100k_v7_frame_stack_1_batch_256_big_net_20210612101118/model_1.pt")

dataset = {
    'observations': [],
    'actions': [], 
    'reward': [], 
    'dones': [],
    'info': []
}

with Xvfb() as xvfb:
    obs = env.reset()
    print(obs.shape)
    for i in range(iterator):
        action = bc.predict([obs.copy()])[0]
        print('action', action)
        obs, done, reward, info = env.step(action)

        dataset['observations'].append(obs)
        dataset['actions'].append(action)
        dataset['reward'].append(reward)
        dataset['dones'].append(done)
        dataset['info'].append(info)


        frame = obs.transpose(2, 1, 0)
        frame_dataset[i] = frame
        print(i)
    
    filename = 'bc_expert_dataset_100k_v2.pkl'
    file = open(os.path.join('/home/ws/ujvhi/d3rlpy/Carla/results_bc/', filename), "wb")
    pickle.dump(dataset, file)
    file.close()

    skvideo.io.vwrite("/home/ws/ujvhi/d3rlpy/Carla/results_bc/bc_expert_dataset_100k_v2_model_1.mp4", videodata = frame_dataset)
