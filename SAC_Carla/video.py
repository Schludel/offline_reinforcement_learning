import imageio
import os
import numpy as np
import skvideo.io


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            try:
                frame = env.render(
                    mode='rgb_array',
                    height=self.height,
                    width=self.width,
                    camera_id=self.camera_id
                )
            except:
                frame = env.render(
                    mode='rgb_array',
                )
    
            #self.frames.append(frame)

    def save(self, file_name, frames_obs):
        if self.enabled:
            self.frames_obs = frames_obs
            print(len(self.frames_obs))
            print(self.frames_obs[0].shape)
            path = os.path.join(self.dir_name, file_name)
            #imageio.mimsave(path, self.frames_obs, fps=self.fps)
            skvideo.io.vwrite(path, videodata = self.frames_obs)

