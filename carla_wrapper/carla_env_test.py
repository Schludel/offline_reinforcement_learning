import gym
import carla_env
from xvfbwrapper import Xvfb

if __name__ == '__main__':
    env = gym.make('CarlaEnv-pixel-v1')

    with Xvfb() as xvfb:

        env.reset()
        done = False
        while not done:
            next_obs, reward, done, info = env.step([1, 0])
            print(done)
        env.close()
