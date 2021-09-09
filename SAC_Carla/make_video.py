import gym 
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import carla_env

import skvideo.io

from curl_sac import RadSacAgent
from curl_sac import CURL

from ResizeObservation import ObservationWrapper
from action_repeat_wrapper import ActionRepeat
from frame_stack_wrapper import FrameStack

env = gym.make('CarlaEnv-pixel-v1')
env = ObservationWrapper(env)


env = FrameStack(env, 4)
env = ActionRepeat(env, 2)

model = RadSacAgent(obs_shape = env.observation_space.shape, 
                    action_shape = env.action_space.shape, 
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    data_augs = 'no_aug',
                    hidden_dim = 1024,
                    )

print(model)
model.load(model_dir = '/home/ws/ujvhi/rad_carla/rad/results/carla_SAC-driving_v70_new_old_rew_temp_0.2_tau_0.005_buffer_100k-07-19-im84-b512-s23-pixel/model', step = '124000')

display_list = []


obs = env.reset()
dones = False
while not dones:
    action = model.select_action(obs / 255.)
    print('action:', action)
    obs, rewards, dones, info = env.step(action)
    env.render()


env.close()



