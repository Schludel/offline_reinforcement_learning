import d3rlpy
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split

from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import continuous_action_diff_scorer
from d3rlpy.metrics.scorer import value_estimation_std_scorer

from ResizeObservation import ObservationWrapper 
from action_repeat_wrapper import ActionRepeat
from frame_stack_wrapper import FrameStack

import gym

from xvfbwrapper import Xvfb

env = gym.make('CarRacing-v0') 
env = ObservationWrapper(env)
env = FrameStack(env, 4)


dataset = MDPDataset.load('./rad/dataset_low_rew/dataset_low_rew_merged_big.h5')

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

bc = d3rlpy.algos.BC(learning_rate=0.0003, use_gpu = True, scaler = 'pixel', batch_size= 512, encoder_factory = 'pixel')

with Xvfb() as xvfb:
    bc.fit(train_episodes,
            eval_episodes = test_episodes,
            n_steps=1000000,
            n_steps_per_epoch=1000,
            logdir = './d3rlpy/CarRacing-v0/d3rlpy_logs/',
            scorers={
                'environment': d3rlpy.metrics.scorer.evaluate_on_environment(env, n_trials=10),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer,
                'value_std': value_estimation_std_scorer,
                'action_diff': d3rlpy.metrics.scorer.continuous_action_diff_scorer,
            },
            tensorboard_dir='./d3rlpy/CarRacing-v0/',
            experiment_name = 'bc_low_reward_dataset_merged_big_high_lr_big_deeper_big_net_with_act_big_batch_v0',
            )
