import d3rlpy
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import continuous_action_diff_scorer
from d3rlpy.metrics.scorer import value_estimation_std_scorer
from ResizeObservation import ObservationWrapper 

import gym
import carla_env

import skvideo.io
import numpy as np
import pickle
import os
from xvfbwrapper import Xvfb



env = gym.make('CarlaEnv-pixel-v1') #turn off autopilot: yes! activate all three spawnpoints, turn off noise
env = ObservationWrapper(env)
dataset = MDPDataset.load('/home/ws/ujvhi/d3rlpy/Carla/expert_dataset_50k_new_reward.h5')

# load full parameters
dqn2 = DQN()

cql = d3rlpy.algos.CQL(use_gpu=True, 
                        scaler = 'pixel', 
                        critic_encoder_factory = 'pixel', 
                        actor_encoder_factory = 'pixel', 
                        n_frames = 4, 
                        batch_size = 256,
                        initial_alpha = 0.01,
                        alpha_learning_rate = 0.0,
                        alpha_threshold = 10.0)

cql.build_with_dataset(dataset)
cql.load_model('/home/ws/ujvhi/d3rlpy/Carla/d3rlpy_logs/expert_dataset_50k_new_reward_alpha_0.01_20210727180518/model_840.pt')

iterator = 500
#video dataset
frame_dataset = np.zeros(shape=(iterator, 84, 84, 3))

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
        action = cql.predict([obs.copy()])[0]
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
    

    skvideo.io.vwrite("/home/ws/ujvhi/d3rlpy/Carla/results_bc/bc_expert_dataset_100k_v2_model_1.mp4", videodata = frame_dataset)
