import d3rlpy

import gym

from d3rlpy.algos import SAC
from d3rlpy.online.buffers import ReplayBuffer
from ResizeObservation import ObservationWrapper
from xvfbwrapper import Xvfb

env = gym.make('CarRacing-v0')
env = ObservationWrapper(env)
eval_env = gym.make('CarRacing-v0')
eval_env = ObservationWrapper(eval_env)

# setup algorithm
sac = SAC(batch_size=50000, use_gpu=False, n_frames = 4, scaler = "pixel")

# replay buffer for experience replay
buffer = ReplayBuffer(maxlen=50000, env=env)

# start training

with Xvfb() as xvfb:
    sac.fit_online(env,
                   buffer,
                   logdir= 'd3rlpy/CarRacing-v0/results/',
                   tensorboard_dir = "d3rlpy/CarRacing-v0/results/",
                   n_steps=100000,
                   eval_env=eval_env,
                   n_steps_per_epoch=1000,
                   update_start_step=1000)