import gym
import numpy as np

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self,env)
        self._height = 96
        self._width = 96
        self.observation_space = gym.spaces.Box(
                    low = 0,
                    high = 255, 
                    shape = (3, self._height, self._width),
                    dtype = np.uint8
                )
    
    def observation(self, obs):
        # modify obs
        #obs = np.dot(obs[...,:3], [0.2989, 0.5870, 0.1140])
        #obs = obs[20:84, 20:84, :]
        #obs = np.reshape(obs, [42, 42, 1])
        #obs = np.reshape(obs, [64, 64, 3])
        obs = obs.transpose(2, 1, 0)
        return obs