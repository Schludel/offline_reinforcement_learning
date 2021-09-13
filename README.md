# Offline Reinforcement Learning CQL
This is the work of my bachelor thesis. Hereby, I focused on the usage of offline reinforcement learning in autonomous driving. To this end, I tested SAC and CQL in the CarRacing-v0 and Carla environment. Furthermore, I created different datasets in the respective environments from expert dataset to noise dataset. I focused on the CQL performance on the different datasets with respect to the alpha value. 

# Installation

## Carla Setup

1. Add the following environment variables:  
```
export PYTHONPATH=$PYTHONPATH:/opt/carla-simulator/PythonAPI
export PYTHONPATH=$PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/
export PYTHONPATH=$PYTHONPATH:/opt/carla-simulator/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg
```
2. Install the following extra libraries  
```
pip install pygame
pip install networkx
pip install dotmap
pip install gym
```

3. Install Carla Wrapper
```
cd carla_wrapper
pip install -e .
```

## Install Gym CarRacing-v0

1. Install the following libraries
```
pip install gym
pip install Box2D
pip install xvfbwrapper
```

## Install SAC

1. Install SAC
```
cd SAC_CarRacing-v0
conda env create -f conda_env.yml
```

## Install d3rlpy (CQL)

1. Install CQL
```
pip install d3rlpy
```

# Usage examples

## SAC CarRacing-v0
1. Train SAC model
```
cd SAC_CarRacing-v0
run bash script/run.sh #edit to evaluate other parameters
```
Models are saved in ./rad/results/

2. Load model
```
import gym 
import numpy as np
from xvfbwrapper import Xvfb
from curl_sac import RadSacAgent

from ResizeObservation import ObservationWrapper
from action_repeat_wrapper import ActionRepeat
from frame_stack_wrapper import FrameStack

env = gym.make('CarRacing-v0')
env = ObservationWrapper(env)

env = FrameStack(env, 4)
env = ActionRepeat(env, 2)

model = RadSacAgent(obs_shape = env.observation_space.shape, 
                    action_shape = env.action_space.shape, 
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                    data_augs = 'no_aug',
                    hidden_dim = 1024,
                    )


model.load(model_dir = './rad/results/CarRacing-v0-driving_test_v21_no_img_crop-06-19-im96-b256-s23-pixel/model', step = '37000')

with Xvfb() as xvfb:
    obs = env.reset()
    dones = False
    while not dones:
        action = model.select_action(obs / 255.)
        print('action:', action)
        obs, rewards, dones, info = env.step(action)
        
        env.render()

env.close()
```

## SAC Carla

1. Start Carla simulator

Open new Terminal and run:
```
bash CarlaUE4.sh -fps 20
```

2. Train SAC model
```
cd SAC_Carla
run bash script/run.sh #edit to evaluate other parameters
```
Models are saved in ./rad/results/

3. Load model
```
import gym 
import numpy as np
from xvfbwrapper import Xvfb
from curl_sac import RadSacAgent

import carla_env

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

model.load(model_dir = './rad/results/carla_SAC-driving_v70_new_old_rew_temp_0.2_tau_0.005_buffer_100k-07-19-im84-b512-s23-pixel/model', step = '124000')

obs = env.reset()
dones = False
while not dones:
    action = model.select_action(obs / 255.)
    obs, rewards, dones, info = env.step(action)
    env.render()
```

## CQL

# CQL CarRacing-v0

1. Train model
```
import d3rlpy
from d3rlpy.dataset import MDPDataset

from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import continuous_action_diff_scorer
from d3rlpy.metrics.scorer import value_estimation_std_scorer

from sklearn.model_selection import train_test_split

from ResizeObservation import ObservationWrapper 
from action_repeat_wrapper import ActionRepeat
from frame_stack_wrapper import FrameStack

import gym

from xvfbwrapper import Xvfb

env = gym.make('CarRacing-v0') 
env = ObservationWrapper(env)
env = FrameStack(env, 4)

#load dataset
dataset = MDPDataset.load('./rad/dataset_high_rew/expert_dataset_merged_big.h5')

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

cql = d3rlpy.algos.CQL(use_gpu=True, 
                        alpha_learning_rate = 0.0, #constant alpha
                        scaler = 'pixel', 
                        critic_encoder_factory = 'pixel', 
                        actor_encoder_factory = 'pixel',  
                        batch_size = 256, 
                        initial_alpha = 0.32,
                        alpha_threshold = 10.0
                        )

with Xvfb() as xvfb:
    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            n_steps=1000000,
            n_steps_per_epoch=1000,
            scorers={
                'environment': d3rlpy.metrics.scorer.evaluate_on_environment(env, 10),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer,
                'value_std': value_estimation_std_scorer,
                'action_diff': continuous_action_diff_scorer,
            },
            tensorboard_dir='./d3rlpy/CarRacing-v0',
            experiment_name = 'poor_dataset_merged_const_alpha_0.32_threshold_10.0',
            )
```

2. Load model

```
import numpy as np
import d3rlpy
from d3rlpy.dataset import MDPDataset

from xvfbwrapper import Xvfb

import gym

from ResizeObservation import ObservationWrapper
from frame_stack_wrapper import FrameStack

# prepare dataset
env = gym.make('CarRacing-v0')
env = ObservationWrapper(env)
env = FrameStack(env, 4)


# prepare algorithm
cql = d3rlpy.algos.CQL(use_gpu=True)

cql.build_with_env(env)

cql.load_model("./d3rlpy/CarRacing-v0/d3rlpy_logs/expert_dataset_merged_learn_alpha_0.01_threshold_10.0_20210720120854/model_57.pt")


obs = env.reset()
done = False
while not done:
    action = cql.predict([obs.copy()])[0]
    obs, reward, done, info = env.step(action)
    env.render()

env.close
```

# CQL Carla

1. Start simulator

Open new Terminal and run:
```
bash CarlaUE4.sh -fps 20
```

2. Train model
```
import d3rlpy
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split

from ResizeObservation import ObservationWrapper 

import gym
import carla_env

env = gym.make('CarlaEnv-pixel-v1')
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
                        alpha_learning_rate = 0.0, #constant alpha
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
```

3. Load model

```

import d3rlpy
from d3rlpy.dataset import MDPDataset

import gym
import carla_env

from ResizeObservation import ObservationWrapper
from frame_stack_wrapper import FrameStack

# prepare dataset
env = gym.make('CarlaEnv-pixel-v1')
env = ObservationWrapper(env)
env = FrameStack(env, 4)

# prepare algorithm
cql = d3rlpy.algos.CQL(use_gpu=True)

cql.build_with_env(env)

cql.load_model("./d3rlpy/Carla/d3rlpy_logs/expert_dataset_50k_new_reward_noise_0.6_alpha_0.1_20210811105023/model_804.pt")



obs = env.reset()
done = False
while not done:
    action = cql.predict([obs.copy()])[0]
    obs, reward, done, info = env.step(action)

env.close
```

# Carla Wrapper

```
cd carla_wrapper/carla_env/__init__.py
```

* edit state-/pixel-based observation
* edit Town
* Turn on/off autopilot




