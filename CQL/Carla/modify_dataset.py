import d3rlpy
from sklearn.model_selection import train_test_split

dataset, env = d3rlpy.datasets.get_d4rl('carla-lane-v0')
train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

cql = d3rlpy.algos.CQL(use_gpu=True, scaler = 'pixel',  n_frames=4 , critic_encoder_factory = 'pixel', actor_encoder_factory = 'pixel')

cql.fit(train_episodes,
        eval_episodes=test_episodes,
        n_steps=1000000,
        n_steps_per_epoch=1000,
        scorers={
            'environment': d3rlpy.metrics.scorer.evaluate_on_environment(env),
            'td_error': d3rlpy.metrics.scorer.td_error_scorer,
            'value_scale': d3rlpy.metrics.scorer.average_value_estimation_scorer
        },
        tensorboard_dir='/home/ws/ujvhi/d3rlpy/Carla/results/',
        experiment_name = 'd4rl_expert_dataset',
        )