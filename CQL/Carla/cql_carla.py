import d3rlpy
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split
from ResizeObservation import ObservationWrapper 

import gym
import carla_env



env = gym.make('CarlaEnv-pixel-v1') #turn off autopilot: yes! activate all three spawnpoints, turn off noise
env = ObservationWrapper(env)
dataset = MDPDataset.load('./d3rlpy/Carla/expert_dataset_50k_new_reward_noise_0.2.h5')

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

cql = d3rlpy.algos.CQL(use_gpu=True, 
                        scaler = 'pixel', 
                        critic_encoder_factory = 'pixel', 
                        actor_encoder_factory = 'pixel', 
                        n_frames = 4, 
                        batch_size = 256,
                        initial_alpha = 0.01,
                        alpha_learning_rate = 0.0,
                        alpha_threshold = 10.0)

cql.fit(train_episodes,
        eval_episodes=test_episodes,
        n_steps=1000000,
        n_steps_per_epoch=500,
        scorers={
            'environment': d3rlpy.metrics.scorer.evaluate_on_environment(env, 5),
        },
        tensorboard_dir='./d3rlpy/Carla/results/',
        experiment_name = 'expert_dataset_50k_new_reward_noise_0.2_alpha_0.01',
        )