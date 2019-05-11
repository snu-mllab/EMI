import argparse
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.atari.atari_env import AtariEnv

import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.optimizers.lbfgs_optimizer import LbfgsOptimizer

from sandbox.rocky.tf.baselines.deterministic_mlp_baseline import DeterministicMLPBaseline
from sandbox.rocky.tf.core.network import ConvNetwork
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy

from rllab.misc.instrument import run_experiment_lite
from rllab.misc import logger

from dsae.dsae import DSAE
from dsae.general_embedding import GeneralEmbedding
from dsae.reconciler import Reconciler
from dsae.trpo_dsae import TRPODSAE
from dsae.embedding_mutualinfo import EmbeddingMutualInfo

import tensorflow as tf

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed', help='RNG seed', type=int, default=2)
parser.add_argument('--n_cpu', type=int, default=int(16))
parser.add_argument('--num_slices', help='Slice big batch into smaller ones to prevent OOM', type=int,
                    default=int(16))
parser.add_argument('--debug', help='debug mode', action='store_true')
parser.add_argument('--log_dir', help='log directory', default=None)
parser.add_argument('--value_function', help='Choose value function baseline', choices=['zero', 'conj', 'adam', 'linear'],
                    default='adam')

parser.add_argument('--n_parallel', type=int, default=int(16))
parser.add_argument('--env', help='environment ID', default='MontezumaRevengeNoFrameskip-v4')
parser.add_argument('--max_path_length', type=int, default=int(4500))
parser.add_argument('--n_itr', type=int, default=int(500))
parser.add_argument('--reward_no_scale', help='Turn off reward scaling', action='store_true')
parser.add_argument('--atari_noop', action='store_true')
parser.add_argument('--atari_eplife', action='store_true')
parser.add_argument('--atari_firereset', action='store_true')
parser.add_argument('--resize_size', type=int, default=int(52))
parser.add_argument('--batch_size', type=int, default=int(100000))
parser.add_argument('--step_size', type=float, default=float(0.01))
parser.add_argument('--discount_factor', type=float, default=float(0.995))

parser.add_argument('--embedding_dim', type=int, default=int(2))
parser.add_argument('--embedding_opt_max_itr', type=int, default=int(3))
parser.add_argument('--actions_unit_gaussian_kl_minimization_loss_weight', type=float, default=5e-1)
parser.add_argument('--replay_pool_size', type=int, default=0)
parser.add_argument('--replay_pool_strategy', type=str, default='subsampled_batch')
parser.add_argument('--residual_method', type=str, default='euclidean')

parser.add_argument('--reconciler_loss_weight', type=float, default=1e2)

parser.add_argument('--residual_ir_coeff', type=float, default=1e-3)
parser.add_argument('--residual_error_ir_normalize', action='store_true')
parser.add_argument('--residual_error_ir_calc_after_opt', action='store_true')
parser.add_argument('--residual_error_ir_use_unnormalized_errors', action='store_true')

parser.add_argument('--mutualinfo_action_loss_weight', type=float, default=1e-1)
parser.add_argument('--mutualinfo_obs_loss_weight', type=float, default=1e-1)

parser.add_argument('--embedding_adam_learning_rate', type=float, default=float(1e-3))

parser.add_argument('--test_trpo_only', action='store_true')

args = parser.parse_args()

def get_value_network(env):
    value_network = ConvNetwork(
        name='value_network',
        input_shape=env.observation_space.shape,
        output_dim=1,
        # number of channels/filters for each conv layer
        conv_filters=(16, 32),
        # filter size
        conv_filter_sizes=(8, 4),
        conv_strides=(4, 2),
        conv_pads=('VALID', 'VALID'),
        hidden_sizes=(256,),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=None,
        batch_normalization=False,
    )
    return value_network

def get_policy_network(env):
    policy_network = ConvNetwork(
        name='prob_network',
        input_shape=env.observation_space.shape,
        output_dim=env.action_space.n,
        # number of channels/filters for each conv layer
        conv_filters=(16, 32),
        # filter size
        conv_filter_sizes=(8, 4),
        conv_strides=(4, 2),
        conv_pads=('VALID', 'VALID'),
        hidden_sizes=(256,),
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.softmax,
        batch_normalization=False,
    )
    return policy_network

def get_policy(env):
    policy_network = get_policy_network(env)
    policy = CategoricalMLPPolicy(
        name='policy',
        env_spec=env.spec,
        prob_network=policy_network
    )
    return policy

def get_baseline(env, value_function, num_slices):
    if (value_function == 'zero'):
        baseline = ZeroBaseline(env.spec)
    else:
        value_network = get_value_network(env)

        if (value_function == 'conj'):
            baseline_optimizer = ConjugateGradientOptimizer(
                subsample_factor=1.0,
                num_slices=num_slices
            )
        elif (value_function == 'adam'):
            baseline_optimizer = FirstOrderOptimizer(
                max_epochs=3,
                batch_size=512,
                num_slices=num_slices,
                ignore_last=True,
                #verbose=True
            )
        else:
            logger.log("Inappropirate value function")
            exit(0)

        baseline = DeterministicMLPBaseline(
            env.spec,
            num_slices=num_slices,
            regressor_args=dict(
                network=value_network,
                optimizer=baseline_optimizer,
                normalize_inputs=False
            )
        )

    return baseline

def get_state_embedding_network_args(env, embedding_dim):
    network_args = dict(
            name='state_embedding_network',
            input_shape=env.observation_space.shape,
            output_dim=embedding_dim,
            conv_filters=(16, 32),
            conv_filter_sizes=(8, 4),
            conv_strides=(4, 2),
            conv_pads=('VALID', 'VALID'),
            hidden_sizes=(256,),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=None,
            batch_normalization=False,
    )
    return network_args

def get_action_embedding_network_args(env, embedding_dim):
    network_args = dict(
            name='action_embedding_network',
            input_shape=(env.action_space.flat_dim,),
            output_dim=embedding_dim,
            hidden_sizes=(64,),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=None,
            batch_normalization=False,
    )
    return network_args

def get_reconciler_common_network_args(env, embedding_dim):
    network_args = dict(
            name='reconciler_common_network',
            output_dim=embedding_dim,
            #hidden_sizes=(64,),
            hidden_sizes=(256,),
            hidden_nonlinearity=tf.nn.relu,
            #hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
            batch_normalization=False,
    )
    return network_args

def get_reconciler_state_network_args(env, embedding_dim):
    network_args = dict(
            name='reconciler_state_network',
            #input_shape=env.observation_space.shape,
            output_dim=None,
            conv_filters=(16, 32),
            conv_filter_sizes=(8, 4),
            conv_strides=(4, 2),
            conv_pads=('VALID', 'VALID'),
            #hidden_sizes=(256,),
            hidden_sizes=(),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=None,
            batch_normalization=False,
    )
    return network_args

def get_reconciler_action_network_args(env, embedding_dim):
    network_args = dict(
            name='reconciler_action_network',
            input_shape=(env.spec.action_space.flat_dim,),
            output_dim=env.spec.action_space.flat_dim,
            #hidden_sizes=(256,),
            hidden_sizes=(),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=None,
            batch_normalization=False,
    )
    return network_args

def get_mutualinfo_action_network_args(env, embedding_dim):
    network_args = dict(
            name='mutualinfo_action_network',
            input_shape=(embedding_dim,),
            output_dim=1,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=None,
            batch_normalization=False,
    )
    return network_args

def get_mutualinfo_obs_network_args(env, embedding_dim):
    network_args = dict(
            name='mutualinfo_obs_network',
            input_shape=(embedding_dim,),
            output_dim=1,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=None,
            batch_normalization=False,
    )
    return network_args

def check_environment():
    required_tf_version = '1.3.0'
    if tf.__version__ != required_tf_version:
        raise Exception('Please use TensorFlow {}'.format(required_tf_version))

def main(_):
  logger.log(str(args))
  logger.log('Main process id: {}'.format(os.getpid()))

  check_environment()

  env = TfEnv(AtariEnv(
      args.env, force_reset=True, record_video=False, record_log=False, resize_size=args.resize_size,
      atari_noop=args.atari_noop, atari_eplife=args.atari_eplife, atari_firereset=args.atari_firereset,
      save_original_frames=False,
  ))

  policy = get_policy(env)
  baseline = get_baseline(env, args.value_function, args.num_slices)

  embedding_dim = args.embedding_dim

  model_args = dict(
          state_network_cls=ConvNetwork,
          state_network_args=get_state_embedding_network_args(env, embedding_dim),
          action_network_cls=MLP,
          action_network_args=get_action_embedding_network_args(env, embedding_dim),
          env_spec=env.spec,
  )

  embeding_optimizer = FirstOrderOptimizer
  embeding_optimizer_args = dict(max_epochs=args.embedding_opt_max_itr, batch_size=512, num_slices=1,
                                 ignore_last=True, learning_rate=args.embedding_adam_learning_rate,
                                 verbose=True)

  replay_pool_size = None
  if (args.replay_pool_size is not None and args.replay_pool_size > 0):
      replay_pool_size = args.replay_pool_size

  if args.reconciler_loss_weight is not None:
      model_args['reconciler_cls'] = Reconciler
      model_args['reconciler_args'] = dict(
              common_network_cls=MLP,
              common_network_args=get_reconciler_common_network_args(env, embedding_dim),
      )
      model_args['reconciler_args']['state_network_cls'] = ConvNetwork
      model_args['reconciler_args']['state_network_args'] = get_reconciler_state_network_args(env, embedding_dim)
      model_args['reconciler_args']['action_network_cls'] = MLP
      model_args['reconciler_args']['action_network_args'] = get_reconciler_action_network_args(env, embedding_dim)

  replay_pool_strategy = args.replay_pool_strategy

  residual_method = args.residual_method

  if args.mutualinfo_action_loss_weight is not None or args.mutualinfo_obs_loss_weight is not None:
      if args.mutualinfo_action_loss_weight is None:
          mutualinfo_action_loss_weight = 0.0
      else:
          mutualinfo_action_loss_weight = args.mutualinfo_action_loss_weight

      if args.mutualinfo_obs_loss_weight is None:
          mutualinfo_obs_loss_weight = 0.0
      else:
          mutualinfo_obs_loss_weight = args.mutualinfo_obs_loss_weight

      model_args['mutualinfo_model_cls'] = EmbeddingMutualInfo
      if mutualinfo_action_loss_weight != 0:
          model_args['mutualinfo_action_model_args'] = dict(
                  network_cls=MLP,
                  network_args=get_mutualinfo_action_network_args(env, embedding_dim*3),
          )

      if mutualinfo_obs_loss_weight != 0:
          model_args['mutualinfo_obs_model_args'] = dict(
                  network_cls=MLP,
                  network_args=get_mutualinfo_obs_network_args(env, embedding_dim*3),
          )

  config = tf.ConfigProto(allow_soft_placement=True,
                          intra_op_parallelism_threads=args.n_cpu,
                          inter_op_parallelism_threads=args.n_cpu)
  config.gpu_options.allow_growth = True  # pylint: disable=E1101
  sess = tf.Session(config=config)
  #with sess.as_default():
  sess.__enter__()

  dsae_args = dict(
        embedding_dim=embedding_dim,
        model_cls=GeneralEmbedding,
        model_args=model_args,
        env_spec=env.spec,
        min_batch_size=args.batch_size,

        use_only_last_frame=True,

        replay_pool_args=dict(
                max_size=replay_pool_size,
                strategy=replay_pool_strategy,
        ),
        optimizer_cls=embeding_optimizer,
        optimizer_args=embeding_optimizer_args,

        residual_method=residual_method,
        actions_unit_gaussian_kl_minimization_loss_weight=args.actions_unit_gaussian_kl_minimization_loss_weight,

        reconciler_loss_weight=args.reconciler_loss_weight,

        residual_error_ir_weight=args.residual_ir_coeff,
        residual_error_ir_normalize=args.residual_error_ir_normalize,
        residual_error_ir_calc_after_opt=args.residual_error_ir_calc_after_opt,
        residual_error_ir_use_unnormalized_errors=args.residual_error_ir_use_unnormalized_errors,
  )

  if args.mutualinfo_action_loss_weight is not None or args.mutualinfo_obs_loss_weight is not None:
      dsae_args['mutualinfo_action_loss_weight'] = mutualinfo_action_loss_weight
      dsae_args['mutualinfo_obs_loss_weight'] = mutualinfo_obs_loss_weight

  dsae = DSAE(**dsae_args)

  if args.test_trpo_only:
      dsae = None

  algo = TRPODSAE(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=args.batch_size,
        max_path_length=args.max_path_length,
        n_itr=args.n_itr,
        discount=args.discount_factor,
        step_size=args.step_size,
        clip_reward=(not args.reward_no_scale),
        optimizer_args={"subsample_factor":1.0,
                        "num_slices":args.num_slices},
        dsae=dsae,
  )
  algo.train(sess)


if __name__ == '__main__':
    if (args.debug):
        main(None)
    else:
        run_experiment_lite(
            main,
            # Number of parallel workers for sampling
            n_parallel=args.n_parallel,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            # Specifies the seed for the experiment. If this is not provided, a random seed
            # will be used
            seed=args.seed,
            log_dir=args.log_dir,
            # plot=True,
        )
