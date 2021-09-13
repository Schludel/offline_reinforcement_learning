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

#Usage examples

##SAC CarRacing-v0
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

##SAC Carla
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





