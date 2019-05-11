import time

import cv2
import numpy as np
import tensorflow as tf

import rllab.misc.logger as logger

USE_TF_PRINT = False

class Noop(object):
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        pass

def tf_print(op, tensors, message=None, summarize=None):
    if not USE_TF_PRINT:
        return op

    stored = []
    for idx, t in enumerate(tensors):
        if not isinstance(t, tf.Tensor):
            stored.append((idx, t))

    for idx, t in reversed(stored):
        del tensors[idx]

    def print_message(*values):
        values = list(values)
        for idx, t in stored:
            values.insert(idx, t)
        logger.log('[' + ']['.join(map(str, values)) + ']')
        return 1.0

    prints = [tf.py_func(print_message, tensors, tf.float64)]
    with tf.control_dependencies(prints):
        op = tf.identity(op)
    return op

class MeasureTime(object):
    def __init__(self, key):
        self._key = key

    def __enter__(self):
        self._time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        logger.record_tabular(self._key, time.time() - self._time)

def take_last_frame_nhwc(obs, state_dim):
    s = obs.shape
    s = (s[0], np.prod(s[1:]))
    assert s[1] % state_dim == 0
    num_channels = s[1] // state_dim
    obs = np.reshape(obs, (s[0], state_dim, num_channels))
    obs = obs[:, :, -1]
    return obs

def convert_space_to_last_frame_only_nhwc(space):
    from sandbox.rocky.tf.spaces.box import Box as TfBox

    low, high = space.bounds
    assert len(low.shape) == 3
    assert len(high.shape) == 3

    low = low[:, :, -1:]
    high = high[:, :, -1:]
    return TfBox(low=low, high=high)

def unstack_stacked_obses(obses, stacked_axis, concat_axis):
    obses = np.asarray(obses)
    num_stack = obses.shape[stacked_axis]
    images = np.split(obses, num_stack, axis=stacked_axis)
    images = np.concatenate(images, axis=concat_axis)
    return images

class ManualEnvSpec(object):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

def convert_env_spec_to_last_frame_only_nhwc(env_spec):
    return ManualEnvSpec(
            observation_space=convert_space_to_last_frame_only_nhwc(env_spec.observation_space),
            action_space=env_spec.action_space)

def flatten_n(x):
    x = np.asarray(x)
    return np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))

def get_dict_identifier(d):
    assert isinstance(d, dict)
    return frozenset(d.items())

def save_image_to_file(path, image):
    if image.dtype != np.uint8:
        image = np.minimum(np.maximum((image / 2.0 + 0.5) * 255.0, 0.0), 255.0)
    cv2.imwrite(path, image)

def get_sorted_indices_for_k_largest_elements(x, k):
    if k <= 0:
        return np.array([], dtype=np.int32)
    indices = np.argpartition(x, -k)[-k:]
    indices = indices[np.argsort(x[indices])]
    indices = np.flip(indices, axis=0)
    return indices

def get_sorted_indices_for_k_smallest_elements(x, k):
    if k <= 0:
        return np.array([], dtype=np.int32)
    indices = np.argpartition(x, k-1)[:k]
    indices = indices[np.argsort(x[indices])]
    return indices

def convert_new_episodes_to_done(new_episodes):
    done = np.concatenate([new_episodes[1:], [True]])
    assert len(new_episodes) == len(done)
    return done

EPSILON = 1e-5

def scale_values(values):
    max_value = np.max(values)
    min_value = np.min(values)

    normalized_values = (values - min_value) / (max_value - min_value + EPSILON)
    return normalized_values

def get_leading_zeros_formatter(num_elements, keyword=''):
    num_digits = len(str(num_elements - 1))
    return '{' + keyword + ':0' + str(num_digits) + 'd}'

