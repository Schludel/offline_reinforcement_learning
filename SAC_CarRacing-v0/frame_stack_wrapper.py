from collections import deque
import numpy as np

from gym.spaces import Box
from gym import Wrapper


class FrameStack(Wrapper):
    def __init__(self,
                 env,
                 num_stack=3):
        super(FrameStack, self).__init__(env)

        self._num_stack = num_stack
        self._frames = deque(maxlen=num_stack)
        low = np.repeat(self.observation_space.low, self._num_stack, axis=0)
        high = np.repeat(self.observation_space.high, self._num_stack, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def _get_observation(self):
        assert len(self._frames) == self._num_stack, (len(self._frames), self._num_stack)
        return np.array(np.concatenate(self._frames))

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._frames.append(list(observation))
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset()
        [self._frames.append(list(observation)) for _ in range(self._num_stack)]
        return self._get_observation()