import d3rlpy
from d3rlpy.dataset import MDPDataset
import h5py

with h5py.File('/home/ws/ujvhi/d3rlpy/Carla/data_collection/expert_dataset_50k_v3_town04.h5', 'r') as f:
    observations = f['observations'][()]
    actions = f['actions'][()]
    rewards = f['rewards'][()]
    terminals = f['terminals'][()]
    episode_terminals = f['episode_terminals'][()]

print('action_len', len(actions))

dataset = MDPDataset.load('/home/ws/ujvhi/d3rlpy/Carla/data_collection/expert_dataset_50k_v3_town04.h5')

stats = dataset.compute_stats()

# reward statistics
print('reward_mean', stats['reward']['mean'])
print('reward_std', stats['reward']['std'])
print('reward_min', stats['reward']['min'])
print('reward_max', stats['reward']['max'])


print('action_mean', stats['action']['mean'])
print('action_std', stats['action']['std'])
print('action_min', stats['action']['min'])
print('action_max', stats['action']['max'])
print('histogram', stats['action']['histogram'])

