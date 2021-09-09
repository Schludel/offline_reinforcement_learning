from gym import Wrapper


class ActionRepeat(Wrapper):
    def __init__(self,
                 env,
                 num_repeat=5):
        super(ActionRepeat, self).__init__(env)
        self._num_repeat = num_repeat

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._num_repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info