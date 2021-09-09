import os
import numpy as np 
import h5py

import d3rlpy
from d3rlpy.dataset import MDPDataset

obs_dataset = []
action_dataset = []
reward_dataset = []
terminal_dataset = []


for i in range(101):
    data_dir = '/home/ws/ujvhi/rad/dataset_low_rew/'
    file_name = 'car_racing_v%d.h5' % i
    path = os.path.join(data_dir, file_name)
    print(path)
    with h5py.File(path, 'r') as f:
        observations = f['observations'][()]
        actions = f['actions'][()]
        rewards = f['rewards'][()]
        terminals = f['terminals'][()]
        print(len(observations))
        print(len(actions))
        print(len(rewards))
        print(len(terminals))
        for j in range(len(observations)):
            obs_dataset.append(observations[j])
            action_dataset.append(actions[j])
            reward_dataset.append(rewards[j])
            terminal_dataset.append(terminals[j])
            
print(sum(reward_dataset)/len(reward_dataset))
obs_dataset = np.array(obs_dataset)
action_dataset = np.array(action_dataset)
reward_dataset = np.array(reward_dataset)
terminal_dataset = np.array(terminal_dataset)
#episode_terminals_dataset = np.array(episode_terminals_dataset)

expert_dataset = MDPDataset(observations = obs_dataset,
                            actions = action_dataset, 
                            rewards = reward_dataset, 
                            terminals = terminal_dataset,
                            #episode_terminals = episode_terminals_dataset, 
                            create_mask = False, 
                            mask_size = 1)

expert_dataset.dump('/home/ws/ujvhi/rad/dataset_low_rew/test_big.h5')