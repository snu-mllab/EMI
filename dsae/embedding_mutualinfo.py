from collections import defaultdict
import copy
import itertools
import numpy as np

from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc.tensor_utils import flatten_tensors
from rllab.misc.tensor_utils import unflatten_tensors
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
import sandbox.rocky.tf.core.layers as L

class EmbeddingMutualInfo(LayersPowered, Serializable):

    def __init__(
            self,
            *,
            network_cls,
            network_args):
        Serializable.quick_init(self, locals())

        logger.log('EmbeddingMutualInfo : {}'.format(locals()))

        network = network_cls(**network_args)
        self._network = network

        output_layers = [self._network.output_layer]
        self.output_layers = output_layers

        LayersPowered.__init__(self, output_layers)

    def compute_output(self, embeddings):
        return L.get_output(self._network.output_layer, {self._network.input_layer: embeddings})