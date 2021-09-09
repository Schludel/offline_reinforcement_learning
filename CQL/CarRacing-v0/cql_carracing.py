import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import PixelEncoderFactory
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
#env = ActionRepeat(env, 2)


#dataset = MDPDataset.load('/home/ws/ujvhi/rad/dataset_low_rew/dataset_low_rew_merged_big.h5')
dataset = MDPDataset.load('/home/ws/ujvhi/rad/dataset_high_rew/expert_dataset_merged_big.h5')

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

#pixel_encoder = PixelEncoderFactory(filters = [(32, 8, 4), (64, 5, 2), (256, 3, 1), (512, 2, 1)], feature_size = 512)
#pixel_encoder = PixelEncoderFactory()
#pixel_encoder.create_with_action([12, 96, 96], 3)

cql = d3rlpy.algos.CQL(use_gpu=True, 
                        alpha_learning_rate = 0.0,
                        scaler = 'pixel', 
                        critic_encoder_factory = 'pixel', 
                        actor_encoder_factory = 'pixel',  
                        batch_size = 256, #######
                        initial_alpha = 0.32,
                        alpha_threshold = 10.0
                        )

with Xvfb() as xvfb:
    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_steps=1000000,
            n_steps_per_epoch=500,
            scorers={
                'environment': d3rlpy.metrics.scorer.evaluate_on_environment(env, 10),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer,
                'value_std': value_estimation_std_scorer,
                'action_diff': continuous_action_diff_scorer,
            },
            tensorboard_dir='/home/ws/ujvhi/d3rlpy/CarRacing-v0',
            #experiment_name = 'expert_dataset_merged_big_alpha_0008_not_deeper_net_with_act_v0',
            #experiment_name = 'expert_dataset_merged_big_alpha_const_09_not_deeper_net_with_act_v0',
            #experiment_name = 'expert_dataset_merged_learn_alpha_0.01_threshold_10.0',
            experiment_name = 'poor_dataset_merged_const_alpha_0.32_threshold_10.0',
            )