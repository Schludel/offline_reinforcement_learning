import gym
import numpy as np

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self,env)
        self._height = 84
        self._width = 84
        self.observation_space = gym.spaces.Box(
                    low = 0,
                    high = 255, 
                    shape = (3, self._height, self._width),
                    dtype = np.uint8
                )
    
    def observation(self, obs):
        # modify obs
        obs = obs.transpose(2, 1, 0)
        return obs