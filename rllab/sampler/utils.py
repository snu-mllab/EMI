import numpy as np
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import logger
from rllab.misc import tensor_utils
import time

def _get_bare_env(env):
    while isinstance(env, ProxyEnv):
        env = env.wrapped_env
    return env

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False, include_original_frames=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    if include_original_frames:
        bare_env = _get_bare_env(env)
        if not hasattr(bare_env, 'get_original_frames'):
            include_original_frames = False
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    next_o = None
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    result = dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),

        last_observation=next_o,
    )
    if include_original_frames:
        original_frames = bare_env.get_original_frames()
        if original_frames is not None:
            result['original_frames'] = original_frames
    return result
