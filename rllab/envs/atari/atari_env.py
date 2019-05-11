import gym
import gym.wrappers
import gym.envs
import gym.spaces
import traceback
import logging

import os
import os.path as osp
from rllab.envs.base import Env, Step
from rllab.envs.gym_env import *
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.misc import logger

from rllab.envs.atari.atari_wrappers import wrap_deepmind, make_atari, get_wrapper_of_specific_type, FrameSaver


class AtariEnv(Env, Serializable):
    def __init__(self, env_name, resize_size=52, atari_noop=True, atari_eplife=False, atari_firereset=False, record_video=True, video_schedule=None, log_dir=None, record_log=True,
                 force_reset=False, save_original_frames=False):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        env = make_atari(env_name, noop=atari_noop)

        env = wrap_deepmind(env=env, resize=resize_size, episode_life=atari_eplife, fire_reset=atari_firereset, save_original_frames=save_original_frames)
        logger.log("resize size: %d" % resize_size)

        self.save_original_frames = save_original_frames

        self.env = env
        self.env_id = env.spec.id

        if self.save_original_frames:
            self.original_frame_saver = get_wrapper_of_specific_type(env, FrameSaver)
            assert self.original_frame_saver is not None

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = convert_gym_space(env.action_space)
        logger.log("action space: {}".format(self._action_space))
        self._horizon = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
        self._log_dir = log_dir
        self._force_reset = force_reset

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        if self._force_reset and self.monitoring:
            from gym.wrappers.monitoring import Monitor
            assert isinstance(self.env, Monitor)
            recorder = self.env.stats_recorder
            if recorder is not None:
                recorder.done = True
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)

    def render(self, mode='human'):
        return self.env.render(mode)

    def terminate(self):
        if self.monitoring:
            self.env._close()
            if self._log_dir is not None:
                print("""
    ***************************
    Training finished! You can upload results to OpenAI Gym by running the following command:
    python scripts/submit_gym.py %s
    ***************************
                """ % self._log_dir)

    def get_original_frames(self):
        if not self.save_original_frames:
            return None
        return self.original_frame_saver.get_frames()

