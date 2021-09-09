import d3rlpy
import gym
import carla_env

from ResizeObservation import ObservationWrapper

env = gym.make('CarlaEnv-pixel-v1') #turn off autopilot?
env = ObservationWrapper(env)

eval_env = gym.make('CarlaEnv-pixel-v1')
eval_env = ObservationWrapper(eval_env)

sac = d3rlpy.algos.SAC(use_gpu=True, scaler = 'pixel', critic_encoder_factory = 'pixel', actor_encoder_factory = 'pixel', n_frames=4, batch_size = 512)

buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen = 50000, env = env)

sac.fit_online(env,
        buffer,
        n_steps=1000000,
        n_steps_per_epoch=1000,
        eval_env = eval_env,
        tensorboard_dir='/home/ws/ujvhi/d3rlpy/Carla/results/',
        experiment_name = 'SAC_frame_stack_4',
        )
                    