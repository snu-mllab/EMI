import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv

def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param

class SparseHalfCheetahEnv(HalfCheetahEnv, Serializable):
    FILE = 'half_cheetah.xml'

    def __init__(self, *args, **kwargs):
        super(SparseHalfCheetahEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self.action_bounds)

        done = False

        body_dist = self.get_body_com("torso")[0]
        if abs(body_dist) <= 5.0:
            reward = 0.
        else:
            reward = 1.0
        return Step(next_obs, reward, done)