from collections import defaultdict
import copy
import itertools
import numpy as np

#from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc.tensor_utils import flatten_tensors
from rllab.misc.tensor_utils import unflatten_tensors
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
import sandbox.rocky.tf.core.layers as L

class Reconciler(Serializable):

    def __init__(
            self,
            *,
            env_spec,
            common_network_cls,
            common_network_args,
            state_input_dim,
            state_network_cls=None,
            state_network_args=dict(),
            action_network_cls=None,
            action_network_args=dict()):
        Serializable.quick_init(self, locals())

        logger.log('Reconciler: {}'.format(locals()))

        self.env_spec = env_spec

        if state_network_cls is not None:
            state_network_args['input_shape'] = env_spec.observation_space.shape
            state_network = state_network_cls(**state_network_args)
            self.state_input_layer = state_network.input_layer
            state_processed_layer = state_network.output_layer
        else:
            self.state_input_layer = L.InputLayer(shape=(None, state_input_dim), input_var=None, name='input_state')
            state_processed_layer = self.state_input_layer

        if action_network_cls is not None:
            action_network_args['input_shape'] = (env_spec.action_space.flat_dim,)
            action_network = action_network_cls(**action_network_args)
            self.action_input_layer = action_network.input_layer
            action_processed_layer = action_network.output_layer
        else:
            self.action_input_layer = L.InputLayer(shape=(None, env_spec.action_space.flat_dim), input_var=None, name='input_action')
            action_processed_layer = self.action_input_layer

        concat_layer = L.concat([L.flatten(state_processed_layer), action_processed_layer])

        common_network_args['input_layer'] = concat_layer
        common_network = common_network_cls(**common_network_args)

        self.output_layer = common_network.output_layer

        self.output_layers = [self.output_layer]

