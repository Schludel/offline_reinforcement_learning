import d3rlpy

from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment

from sklearn.model_selection import train_test_split

dataset, env = d3rlpy.datasets.get_d4rl('carla-lane-v0')

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

cql = d3rlpy.algos.CQL(use_gpu=True)
cql.build_with_dataset(dataset)

# set environment in scorer function
evaluate_scorer = evaluate_on_environment(env)
print('evaluate_scorer', evaluate_scorer)

# evaluate algorithm on the environment
rewards = evaluate_scorer(cql)
print('rewards', rewards)