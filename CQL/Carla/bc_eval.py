import d3rlpy
from d3rlpy.dataset import MDPDataset

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


dataset = MDPDataset.load('./d3rlpy/Carla/data_collection/expert_dataset_100k.h5')

iterator = 500

# prepare algorithm
bc = d3rlpy.algos.BC(use_gpu=True)

bc.build_with_dataset(dataset)

bc.load_model("./d3rlpy/Carla/results_bc/bc_expert_dataset_100k_v7_frame_stack_1_batch_256_big_net_20210612101118/model_1.pt")


with Xvfb() as xvfb:
    obs = env.reset()
    for i in range(iterator):
        action = bc.predict([obs.copy()])[0]
        obs, done, reward, info = env.step(action)
    
