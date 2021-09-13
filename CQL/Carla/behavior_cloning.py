import d3rlpy
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split
from ResizeObservation import ObservationWrapper 

from ResizeObservation import ObservationWrapper 
from frame_stack_wrapper import FrameStack

import gym

from xvfbwrapper import Xvfb


import gym
import carla_env

env = gym.make('CarlaEnv-pixel-v1')
env = ObservationWrapper(env)


dataset = MDPDataset.load('./d3rlpy/Carla/data_collection/expert_dataset_50k_new_reward_noise_0.8.h5')

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

bc = d3rlpy.algos.BC(learning_rate=0.0003, use_gpu = True, scaler = 'pixel', batch_size= 512, encoder_factory = 'pixel', n_frames=4)

bc.fit(train_episodes,
        eval_episodes = test_episodes,
        n_steps=1000000,
        n_steps_per_epoch=1000,
        logdir = './d3rlpy/Carla/results_bc/',
        scorers={
            'environment': d3rlpy.metrics.scorer.evaluate_on_environment(env, n_trials=5),
        },
        tensorboard_dir='./d3rlpy/Carla/results_bc/',
        experiment_name = 'bc_expert_dataset_new_reward_function_noise_0.8_v1',
        )
