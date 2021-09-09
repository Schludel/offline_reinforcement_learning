import d3rlpy
import gym
from ResizeObservation import ObservationWrapper

from xvfbwrapper import Xvfb

env = gym.make('CarRacing-v0')
env = ObservationWrapper(env)

# prepare algorithm
sac = d3rlpy.algos.SAC(batch_size=50000, use_gpu=False, n_frames = 4, scaler = "pixel")

sac.build_with_env(env)

sac.load_model("/home/ws/ujvhi/d3rlpy/CarRacing-v0/d3rlpy/CarRacing-v0/results/SAC_online_20210602145957/model_44000.pt")

with Xvfb() as xvfb:
    obs = env.reset()
    while(True):
        action = sac.predict([obs.copy])[0]
        print('action', action)
        obs, done, reward, info = env.step(action)
        print('reward', reward)