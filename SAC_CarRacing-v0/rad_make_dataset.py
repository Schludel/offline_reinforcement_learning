import gym 
import numpy as np

import d3rlpy
from d3rlpy.dataset import MDPDataset

from xvfbwrapper import Xvfb
import skvideo.io

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

obs_dataset = []
action_dataset = []
reward_dataset = []
terminal_dataset = []


with Xvfb() as xvfb:
    obs = env.reset()
    dones = False
    while not dones:
        action = model.select_action(obs / 255.)
        print('action:', action)
        obs, rewards, dones, info = env.step(action)

        obs_dataset.append(obs)
        action_dataset.append(action)
        reward_dataset.append(rewards)
        terminal_dataset.append(dones)

        env.render()

print('TOTAL REWARD:', sum(reward_dataset))


obs_dataset = np.array(obs_dataset)
obs_dataset = obs_dataset[10:]
action_dataset = np.array(action_dataset)
action_dataset = action_dataset[10:]
reward_dataset = np.array(reward_dataset)
reward_dataset = reward_dataset[10:]
terminal_dataset = np.array(terminal_dataset)
terminal_dataset = terminal_dataset[10:]


expert_dataset = MDPDataset(observations = obs_dataset,
                            actions = action_dataset, 
                            rewards = reward_dataset, 
                            terminals = terminal_dataset,
                            #episode_terminals = episode_terminals, 
                            create_mask = False, 
                            mask_size = 1)

expert_dataset.dump('./rad/dataset_low_rew/car_racing_v100.h5')

env.close()