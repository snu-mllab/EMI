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

class GeneralEmbedding(LayersPowered, Serializable):

    def __init__(
            self,
            *,
            state_network_cls,
            state_network_args,
            action_network_cls,
            action_network_args,
            env_spec,
            reconciler_cls=None,
            reconciler_args=dict(),
            mutualinfo_model_cls=None,
            mutualinfo_action_model_args=None,
            mutualinfo_obs_model_args=None
    ):
        """Use given state embedding network and one FC for action embedding."""
        Serializable.quick_init(self, locals())

        logger.log('GeneralEmbedding: {}'.format(locals()))

        self.embedding_dim = state_network_args['output_dim']
        self.env_spec = env_spec

        state_network = state_network_cls(**state_network_args)
        self._state_network = state_network
        self._l_state = state_network.input_layer
        self._l_phi = state_network.output_layer

        action_network = action_network_cls(**action_network_args)
        self.action_network = action_network
        self._l_action = action_network.input_layer
        self._l_psi = action_network.output_layer

        output_layers = [self._l_phi, self._l_psi]

        if reconciler_cls is not None:
            reconciler_state_input_dim = env_spec.observation_space.flat_dim
            reconciler_args['state_input_dim'] = reconciler_state_input_dim
            reconciler_args['common_network_args']['input_shape'] = (reconciler_state_input_dim + env_spec.action_space.flat_dim,)
            reconciler_args['env_spec'] = env_spec

            self.reconciler = reconciler_cls(**reconciler_args)
            output_layers.extend(self.reconciler.output_layers)
        else:
            self.reconciler = None

        if mutualinfo_model_cls is not None:
            if mutualinfo_action_model_args is not None:
                self.mutualinfo_action_model = mutualinfo_model_cls(**mutualinfo_action_model_args)
                output_layers.extend(self.mutualinfo_action_model.output_layers)
            else:
                self.mutualinfo_action_model = None
            if mutualinfo_obs_model_args is not None:
                self.mutualinfo_obs_model = mutualinfo_model_cls(**mutualinfo_obs_model_args)
                output_layers.extend(self.mutualinfo_obs_model.output_layers)
            else:
                self.mutualinfo_obs_model = None



        LayersPowered.__init__(self, output_layers)

        phi_output = L.get_output(self._l_phi)
        psi_output = L.get_output(self._l_psi)

        self._obs_to_phi = tensor_utils.compile_function(
                inputs=[self._l_state.input_var],
                outputs=phi_output)
        self._action_to_psi = tensor_utils.compile_function(
                inputs=[self._l_action.input_var],
                outputs=psi_output)

        if self.reconciler is not None:
            self._obs_action_to_reconciler = tensor_utils.compile_function(
                    inputs=[self.reconciler.state_input_layer.input_var, self.reconciler.action_input_layer.input_var],
                    outputs=L.get_output(self.reconciler.output_layer))

    def compute_embeddings_given_state_action_pairs(self, obses, actions):
        result = dict()
        actions = tf.cast(actions, tf.float32)
        phi = L.get_output(self._l_phi, {self._l_state: obses})
        if self.reconciler is not None:
            reconciler_state_input = obses
        result['phi'] = phi
        result['psi'] = L.get_output(self._l_psi, {self._l_action: actions})
        if self.reconciler is not None:
            result['reconciler'] = L.get_output(
                    self.reconciler.output_layer,
                    {
                        self.reconciler.state_input_layer: reconciler_state_input,
                        self.reconciler.action_input_layer: actions,
                    }
            )
        return result

    def compute_state_embeddings(self, obses):
        return L.get_output(self._l_phi, {self._l_state: obses})

    def compute_action_embeddings(self, actions):
        return L.get_output(self._l_psi, {self._l_action: tf.cast(actions, tf.float32)})

    def eval_state_embeddings(self, obses):
        return self._obs_to_phi(obses)

    def eval_action_embeddings(self, actions):
        return self._action_to_psi(actions)

    def eval_reconcilers(self, obses, actions):
        assert self.reconciler is not None
        return self._obs_action_to_reconciler(obses, actions)


