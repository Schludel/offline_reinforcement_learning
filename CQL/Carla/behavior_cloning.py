import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import PixelEncoderFactory
from sklearn.model_selection import train_test_split
from ResizeObservation import ObservationWrapper 

from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import continuous_action_diff_scorer
from d3rlpy.metrics.scorer import value_estimation_std_scorer

from ResizeObservation import ObservationWrapper 
from frame_stack_wrapper import FrameStack

import gym

from xvfbwrapper import Xvfb


import gym
import carla_env

env = gym.make('CarlaEnv-pixel-v1') #turn off autopilot? activate spawnpoints 
env = ObservationWrapper(env)
#env = FrameStack(env, 4)

dataset = MDPDataset.load('/home/ws/ujvhi/d3rlpy/Carla/data_collection/expert_dataset_50k_new_reward_noise_0.8.h5')

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

#pixel_encoder = PixelEncoderFactory(filters = [(32, 8, 5), (64, 5, 2), (128, 4, 2), (256, 2, 1), (512, 1, 1)], feature_size=256)
#pixel_encoder.create_with_action([12, 84, 84], 2)

bc = d3rlpy.algos.BC(learning_rate=0.0003, use_gpu = True, scaler = 'pixel', batch_size= 512, encoder_factory = 'pixel', n_frames=4) #encoder factory is pixel!

bc.fit(train_episodes,
        eval_episodes = test_episodes,
        n_steps=1000000,
        n_steps_per_epoch=1000,
        logdir = '/home/ws/ujvhi/d3rlpy/Carla/results_bc/',
        scorers={
            'environment': d3rlpy.metrics.scorer.evaluate_on_environment(env, n_trials=5),
        },
        tensorboard_dir='/home/ws/ujvhi/d3rlpy/Carla/results_bc/',
        experiment_name = 'bc_expert_dataset_new_reward_function_noise_0.8_v1',
        )
