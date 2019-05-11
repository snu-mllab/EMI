import numpy as np
import tensorflow as tf

from rllab.core.serializable import Serializable
import rllab.misc.logger as logger
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.optimizers.first_order_optimizer import FirstOrderOptimizer
from sandbox.rocky.tf.spaces.box import Box

from dsae.calc_utils import compute_sample_covariance, multivariate_kl_with_unit_gaussian_sym
from dsae.replay_pool import ReplayPool
from dsae.similarity_calculator import SimilarityCalculator
from dsae.utils import tf_print, MeasureTime, take_last_frame_nhwc, convert_env_spec_to_last_frame_only_nhwc, flatten_n, scale_values

class DSAE(Serializable):
    def __init__(self,
                 *,
                 embedding_dim,
                 model_cls,
                 model_args,
                 env_spec,
                 min_batch_size,

                 clip_actions=True,
                 use_only_last_frame=False,

                 replay_pool_args=dict(),
                 optimizer_cls=FirstOrderOptimizer,
                 optimizer_args=dict(max_opt_itr=100),

                 residual_method='euclidean',
                 actions_unit_gaussian_kl_minimization_loss_weight=None,

                 reconciler_loss_weight=None,

                 mutualinfo_action_loss_weight=None,
                 mutualinfo_obs_loss_weight=None,
                 eval_chunk_size=512,

                 diversity_seeking_ir_weight=None,
                 diversity_seeking_calc='relative',
                 diversity_seeking_kernel_bandwidth=5.0,
                 diversity_seeking_pool='replay_pool',
                 compute_similarities_on_gpu=True,
                 compute_similarities_on_gpu_chunk_size=256,

                 residual_error_ir_weight=None,
                 residual_error_ir_normalize=False,
                 residual_error_ir_calc_after_opt=False,
                 residual_error_ir_use_unnormalized_errors=False):
        Serializable.quick_init(self, locals())

        logger.log('DSAE: {}'.format(locals()))

        self.embedding_dim = embedding_dim

        self.env_spec = env_spec
        self.env_spec_for_embedding = env_spec
        self.clip_actions = clip_actions
        self.use_only_last_frame = use_only_last_frame

        if self.use_only_last_frame:
            self.env_spec_for_embedding = convert_env_spec_to_last_frame_only_nhwc(env_spec)

            self.state_dim = np.prod(self.env_spec.observation_space.shape[:-1])

            state_input_shape = model_args['state_network_args']['input_shape']
            state_input_shape = (tuple(state_input_shape[:-1]) + (1,))
            model_args['state_network_args']['input_shape'] = state_input_shape

        model_args['env_spec'] = self.env_spec_for_embedding

        self.replay_pool = ReplayPool(
                observation_space=self.env_spec_for_embedding.observation_space,
                action_space=self.env_spec_for_embedding.action_space,
                min_batch_size=min_batch_size,
                extra_shapes=[()],  # new_episodes
                extra_dtypes=[bool],  # new_episodes
                **replay_pool_args)
        self.optimizer = optimizer_cls(
                name='embedding',
                **optimizer_args)

        self.residual_method = residual_method
        self.actions_unit_gaussian_kl_minimization_loss_weight = actions_unit_gaussian_kl_minimization_loss_weight

        self.mutualinfo_action_loss_weight = mutualinfo_action_loss_weight
        self.mutualinfo_obs_loss_weight = mutualinfo_obs_loss_weight
        assert (self.mutualinfo_action_loss_weight is not None) == (self.mutualinfo_obs_loss_weight is not None)

        self.eval_chunk_size = eval_chunk_size

        self.diversity_seeking_ir_weight = diversity_seeking_ir_weight
        self.diversity_seeking_calc = diversity_seeking_calc
        self.diversity_seeking_kernel_bandwidth = diversity_seeking_kernel_bandwidth
        self.diversity_seeking_pool = diversity_seeking_pool
        self.compute_similarities_on_gpu = compute_similarities_on_gpu
        if self.compute_similarities_on_gpu:
            self.similarity_calculator = SimilarityCalculator(
                    chunk_size=compute_similarities_on_gpu_chunk_size)

        self.residual_error_ir_weight = residual_error_ir_weight
        self.residual_error_ir_normalize = residual_error_ir_normalize
        self.residual_error_ir_calc_after_opt = residual_error_ir_calc_after_opt
        self.residual_error_ir_use_unnormalized_errors = residual_error_ir_use_unnormalized_errors

        self.model = model_cls(**model_args)

        self.reconciler_loss_weight = reconciler_loss_weight
        assert (reconciler_loss_weight is not None) == (self.model.reconciler is not None)

        self._init_opt()


    def _init_opt(self):
        embedding_optimizer_input_list = []
        def _register_variable(var):
            embedding_optimizer_input_list.append(var)
            return var

        next_obses = _register_variable(self.env_spec_for_embedding.observation_space.new_tensor_variable(
                'next_obses_embedding',
                extra_dims=1,
                flatten=True,
        ))
        obses = _register_variable(self.env_spec_for_embedding.observation_space.new_tensor_variable(
                'obses_embedding',
                extra_dims=1,
                flatten=True,
        ))
        actions = _register_variable(self.env_spec_for_embedding.action_space.new_tensor_variable(
                'action_embedding',
                extra_dims=1,
        ))

        batch_size = tf.cast(tf.shape(next_obses)[0], tf.float32)

        embedding_result = self.model.compute_embeddings_given_state_action_pairs(obses, actions)
        phi_t = embedding_result['phi']
        psi_t = embedding_result['psi']
        if self.reconciler_loss_weight is not None:
            reconciler_t = embedding_result['reconciler']
            final_reconciler_t = reconciler_t
        else:
            final_reconciler_t = None
        phi_t_plus_one = self.model.compute_state_embeddings(next_obses)

        result = self._construct_error_vectors(
                phi_t, phi_t_plus_one, psi_t, reconciler=final_reconciler_t)
        error_vectors = result['error_vectors']
        phi_diffs = result['phi_diffs']
        pure_error_vectors = result['pure_error_vectors']

        res = tf.reduce_sum(tf.square(error_vectors), axis=1)

        del result

        phi_diff_norms = tf.reduce_sum(tf.square(phi_diffs), axis=1)

        if self.residual_method == 'euclidean':
            loss_residual = tf.reduce_mean(res)
        else:
            assert False

        embedding_loss = loss_residual

        if self.reconciler_loss_weight is not None:
            pure_res = tf.reduce_sum(tf.square(pure_error_vectors), axis=1)

            reconciler_pure_norms = tf.reduce_sum(tf.square(final_reconciler_t), axis=1)
            reconciler_norms = reconciler_pure_norms
            loss_reconciler = tf.reduce_mean(reconciler_norms)

            embedding_loss += self.reconciler_loss_weight * loss_reconciler

        half_batch_size = tf.cast(tf.divide(batch_size, 2), tf.int32)
        if self.mutualinfo_action_loss_weight is not None:
            if self.mutualinfo_action_loss_weight != 0:
                action_joint_output = self.model.mutualinfo_action_model.compute_output(
                        tf.concat([psi_t[:half_batch_size],
                            phi_t_plus_one[:half_batch_size],
                            phi_t[:half_batch_size]], axis=-1))
                action_marginal_output = self.model.mutualinfo_action_model.compute_output(
                        tf.concat([psi_t[half_batch_size:2*half_batch_size],
                            phi_t_plus_one[:half_batch_size],
                            phi_t[:half_batch_size]], axis=-1))
                action_mutual_loss = tf.reduce_mean(
                    tf.nn.softplus(tf.negative(action_joint_output)) + tf.nn.softplus(action_marginal_output))
            else:
                action_mutual_loss = tf.constant(0.0)
            if self.mutualinfo_obs_loss_weight != 0:
                obs_joint_output = self.model.mutualinfo_obs_model.compute_output(
                        tf.concat([phi_t_plus_one[:half_batch_size],
                            phi_t[:half_batch_size],
                            psi_t[:half_batch_size]], axis=-1))
                obs_marginal_output = self.model.mutualinfo_obs_model.compute_output(
                        tf.concat([phi_t_plus_one[half_batch_size:2*half_batch_size],
                            phi_t[:half_batch_size],
                            psi_t[:half_batch_size]], axis=-1))
                obs_mutual_loss = tf.reduce_mean(
                    tf.nn.softplus(tf.negative(obs_joint_output)) + tf.nn.softplus(obs_marginal_output))
            else:
                obs_mutual_loss = tf.constant(0.0)

            mutual_loss = self.mutualinfo_action_loss_weight * action_mutual_loss + self.mutualinfo_obs_loss_weight * obs_mutual_loss
            embedding_loss += mutual_loss


        if self.actions_unit_gaussian_kl_minimization_loss_weight is not None:
            psi_t_mean = tf.reduce_mean(psi_t, axis=0)
            psi_t_shifted = psi_t - psi_t_mean
            psi_t_covariance = compute_sample_covariance(psi_t_shifted, batch_size, 'actions')
            actions_unit_gaussian_kl = multivariate_kl_with_unit_gaussian_sym(
                    psi_t_mean, psi_t_covariance, self.embedding_dim)

            embedding_loss += self.actions_unit_gaussian_kl_minimization_loss_weight * actions_unit_gaussian_kl


        update_opt_args = dict(
                loss=embedding_loss,
                target=self.model,
                inputs=embedding_optimizer_input_list,
        )
        self.optimizer.update_opt(**update_opt_args)


        self._eval_loss_residual = tensor_utils.compile_function(
                inputs=embedding_optimizer_input_list, outputs=loss_residual)

        main_outputs = [
            phi_t,
            phi_t_plus_one,
            psi_t,
            res,
            phi_diff_norms,
        ]
        if self.reconciler_loss_weight is not None:
            main_outputs.append(pure_res)
            main_outputs.append(final_reconciler_t)
            main_outputs.append(reconciler_norms)
            main_outputs.append(reconciler_pure_norms)

        inference_only_main_input_list = [
            next_obses,
            obses,
            actions,
        ]
        reconciler_input_list = [
            obses,
            actions,
        ]

        self._inference_only_inputs_assigner = None
        self._reconciler_inputs_assigner = None

        self._eval_main_outputs = tensor_utils.compile_function(
                inputs=inference_only_main_input_list,
                outputs=main_outputs,
        )

        self._eval_reconcilers = tensor_utils.compile_function(
                inputs=reconciler_input_list,
                outputs=[final_reconciler_t],
        )

        self._eval_residual_errors = tensor_utils.compile_function(
                inputs=inference_only_main_input_list,
                outputs=[res])

        all_metrics = [
            embedding_loss,
            loss_residual,
        ]
        if self.reconciler_loss_weight is not None:
            all_metrics.append(loss_reconciler)
        if self.actions_unit_gaussian_kl_minimization_loss_weight is not None:
            all_metrics.append(actions_unit_gaussian_kl)
        if self.mutualinfo_action_loss_weight is not None:
            all_metrics.append(mutual_loss)
            all_metrics.append(action_mutual_loss)
            all_metrics.append(obs_mutual_loss)
        self._eval_metrics = tensor_utils.compile_function(
                inputs=embedding_optimizer_input_list, outputs=all_metrics)

        if self.reconciler_loss_weight is not None:
            self._eval_loss_reconciler = tensor_utils.compile_function(
                    inputs=embedding_optimizer_input_list, outputs=loss_reconciler)

        if self.actions_unit_gaussian_kl_minimization_loss_weight is not None:
            self._eval_actions_unit_gaussian_kl_loss = tensor_utils.compile_function(
                    inputs=embedding_optimizer_input_list, outputs=actions_unit_gaussian_kl)

    def _construct_error_vectors(self, phi_t, phi_t_plus_one, psi_t, reconciler=None):
        result = dict()

        phi_diffs = phi_t_plus_one - phi_t
        result['phi_diffs'] = phi_diffs

        result['pure_error_vectors'] = phi_diffs - psi_t

        if self.reconciler_loss_weight is not None:
            assert reconciler is not None
            assert reconciler.get_shape().as_list()[-1] == self.embedding_dim
            result['error_vectors'] = result['pure_error_vectors'] - reconciler
        else:
            result['error_vectors'] = result['pure_error_vectors']

        return result

    def _preprocess_obs(self, obs):
        if self.use_only_last_frame:
            obs = take_last_frame_nhwc(obs, self.state_dim)
        obs = flatten_n(obs)
        return obs

    def train_and_compute_intrinsic_rewards(self, itr, paths):
        logger.log('DSAE.train_and_compute_intrinsic_rewards')
        with MeasureTime('EmbeddingPreprocessSamplesTime'):
            observations_raw = np.concatenate([self.env_spec.observation_space.unflatten_n(p['observations']) for p in paths])
            observations = self._preprocess_obs(observations_raw)

            concat_targets = []
            for p in paths:
                concat_targets.append(self.env_spec.observation_space.unflatten_n(p['observations'][1:]))
                concat_targets.append(p['last_observation'][np.newaxis])
            next_observations_raw = np.concatenate(concat_targets, axis=0)
            next_observations = self._preprocess_obs(next_observations_raw)

            actions = np.concatenate([p['actions'] for p in paths])
            if self.clip_actions:
                if isinstance(self.env_spec_for_embedding.action_space, Box):
                    action_low, action_high = self.env_spec_for_embedding.action_space.bounds
                    actions = np.clip(actions, action_low, action_high)
                    logger.log('Actions clipped for embedding')

            logger.log('next_observations: {}, actions: {}'.format(next_observations.shape, actions.shape))

            path_lengths = [len(p['rewards']) for p in paths]
            new_episodes = np.zeros((sum(path_lengths),), dtype=np.bool)
            last_position = 0
            path_length_cum_sum = [last_position]
            for l in path_lengths:
                new_episodes[last_position] = True
                last_position += l
                path_length_cum_sum.append(last_position)
            assert last_position == len(new_episodes)
            assert last_position == len(observations)

            env_infos = tensor_utils.concat_tensor_dict_list(
                    [path['env_infos'] for path in paths])

        residual_errors_before = self._get_general_eval_result(
                eval_func=self._eval_residual_errors,
                assigner_func=self._inference_only_inputs_assigner,
                all_input_values=[next_observations, observations, actions],
                chunk_size=self.eval_chunk_size)[0]

        unnormalized_residual_errors_before = None

        self.replay_pool.add_samples(
                obses=observations,
                next_obses=next_observations,
                actions=actions,
                extras=[new_episodes])

        self._optimize(itr)

        with MeasureTime('EmbeddingEvalMainOutputsTime'):
            main_outputs = self._get_general_eval_result(
                    eval_func=self._eval_main_outputs,
                    assigner_func=self._inference_only_inputs_assigner,
                    all_input_values=[next_observations, observations, actions],
                    chunk_size=self.eval_chunk_size)

        obs_embeddings = main_outputs.pop(0)
        next_obs_embeddings = main_outputs.pop(0)
        action_embeddings = main_outputs.pop(0)
        residual_errors_after = main_outputs.pop(0)
        phi_diff_norms = main_outputs.pop(0)

        unnormalized_residual_errors_after = None
        if self.reconciler_loss_weight is not None:
            pure_residual_errors_after = main_outputs.pop(0)

            reconcilers = main_outputs.pop(0)

            reconciler_norms = main_outputs.pop(0)
            logger.record_tabular('ReconcilerNormMax', np.max(reconciler_norms))
            logger.record_tabular('ReconcilerNormMin', np.min(reconciler_norms))
            logger.record_tabular('ReconcilerNormMean', np.mean(reconciler_norms))

            reconciler_pure_norms = main_outputs.pop(0)
            logger.record_tabular('ReconcilerPureNormMax', np.max(reconciler_pure_norms))
            logger.record_tabular('ReconcilerPureNormMin', np.min(reconciler_pure_norms))
            logger.record_tabular('ReconcilerPureNormMean', np.mean(reconciler_pure_norms))

            reconciler_scalers = None
        else:
            pure_residual_errors_after = None
            reconcilers = None
            reconciler_norms = None
            reconciler_pure_norms = None
            reconciler_scalers = None

        assert len(main_outputs) == 0

        returns_orig = np.asarray([np.sum(path["raw_rewards"]) for path in paths])

        with MeasureTime('EmbeddingComputeIntrinsicRewardsTime'):
            intrinsic_rewards = self._compute_intrinsic_rewards(
                    obses=observations,
                    next_obses=next_observations,
                    actions=actions,
                    obs_embeddings=obs_embeddings,
                    next_obs_embeddings=next_obs_embeddings,
                    action_embeddings=action_embeddings,
                    new_episodes=new_episodes,
                    residual_errors_before=residual_errors_before,
                    residual_errors_after=residual_errors_after,
                    unnormalized_residual_errors_before=unnormalized_residual_errors_before,
                    unnormalized_residual_errors_after=unnormalized_residual_errors_after,
                    reconcilers=reconcilers,
                    reconciler_norms=reconciler_norms)
        assert len(intrinsic_rewards) == last_position

        last_position = 0
        for p in paths:
            l = len(p['rewards'])
            p['rewards'] = p['rewards'] + intrinsic_rewards[last_position:last_position+l]
            last_position += l

    def _get_general_eval_result(self, 
                                 *,
                                 eval_func,
                                 assigner_func,
                                 all_input_values,
                                 chunk_size=None):
        total_size = len(all_input_values[0])
        if chunk_size is None:
            chunk_size = total_size
        results = []
        for start in range(0, total_size, chunk_size):
            input_values = [v[start:start + chunk_size] for v in all_input_values]
            results.append(eval_func(*input_values))

        outputs = []
        for chunked_outputs in zip(*results):
            outputs.append(np.concatenate(chunked_outputs, axis=0))

        return outputs

    def _get_eval_result(self, 
                         *,
                         eval_func,
                         next_observations,
                         observations,
                         actions,
                         new_episodes):
        all_input_values = (
            next_observations,
            observations,
            actions,
        )

        result = eval_func(*all_input_values)

        return result

    def _optimize(self, itr):
        ordered = None
        all_input_values = self.replay_pool.get_data(ordered=ordered)
        assert len(all_input_values) == 3

        with MeasureTime('EmbeddingObtainMetricBeforeTime'):
            metrics_before_opt = self._obtain_metric_set(all_input_values)

        logger.log('Optimizing embedding')
        with MeasureTime('EmbeddingOptTime'):
            info_dict = self.optimizer.optimize(all_input_values)
        if info_dict is not None:
            logger.log('Embedding optimization info ::: nit: {}, warnflag: {}, funcalls: {}, task: {}, grad: {}'.format(
                    info_dict['nit'], info_dict['warnflag'], info_dict['funcalls'],
                    info_dict.get('task', None), info_dict['grad']))
            logger.record_tabular('NumEmbeddingOptIter', info_dict['nit'])
        with MeasureTime('EmbeddingObtainMetricAfterTime'):
            metrics_after_opt = self._obtain_metric_set(all_input_values)

        metric_keys = [
            'EmbeddingLoss',
            'EmbeddingLossResidual',
        ]
        if self.reconciler_loss_weight is not None:
            metric_keys.append('EmbeddingLossReconciler')
        if self.actions_unit_gaussian_kl_minimization_loss_weight is not None:
            metric_keys.append('EmbeddingActionsUnitGaussianKLLoss')
        if self.mutualinfo_action_loss_weight is not None:
            metric_keys.append('EmbeddingMutualInfoLoss')
            metric_keys.append('EmbeddingActionMutualInfoLoss')
            metric_keys.append('EmbeddingObsMutualInfoLoss')

        metrics_diff = self._obtain_diff_metric_set(metric_keys, metrics_before_opt, metrics_after_opt)

        self._log_metrics(metrics_before_opt, 'Before')
        self._log_metrics(metrics_after_opt, 'After')
        self._log_metrics(metrics_diff, 'Diff')

    def _obtain_metric_set(self, all_input_values):
        all_metrics = self._eval_metrics(*(all_input_values))

        metric_set = dict()
        metric_set['EmbeddingLoss'] = all_metrics.pop(0)
        metric_set['EmbeddingLossResidual'] = all_metrics.pop(0)
        if self.reconciler_loss_weight is not None:
            metric_set['EmbeddingLossReconciler'] = all_metrics.pop(0)
        if self.actions_unit_gaussian_kl_minimization_loss_weight is not None:
            metric_set['EmbeddingActionsUnitGaussianKLLoss'] = all_metrics.pop(0)
        if self.mutualinfo_action_loss_weight is not None:
            metric_set['EmbeddingMutualInfoLoss'] = all_metrics.pop(0)
            metric_set['EmbeddingActionMutualInfoLoss'] = all_metrics.pop(0)
            metric_set['EmbeddingObsMutualInfoLoss'] = all_metrics.pop(0)

        assert len(all_metrics) == 0

        return metric_set

    def _obtain_diff_metric_set(self, keys, metrics_before_opt, metrics_after_opt):
        metric_set = dict()
        for k in keys:
            metric_set[k] = metrics_before_opt[k] - metrics_after_opt[k]
        return metric_set

    def _log_metrics(self, metric_set, tag):
        for k, v in metric_set.items():
            logger.record_tabular('{} {}'.format(k, tag), v)

    def _compute_intrinsic_rewards(self,
                                   *,
                                   obses,
                                   next_obses,
                                   actions,
                                   obs_embeddings,
                                   next_obs_embeddings,
                                   action_embeddings,
                                   new_episodes,
                                   residual_errors_before,
                                   residual_errors_after,
                                   unnormalized_residual_errors_before,
                                   unnormalized_residual_errors_after,
                                   reconcilers,
                                   reconciler_norms):
        all_irs = np.zeros((len(obs_embeddings),))

        if self.diversity_seeking_ir_weight is not None:
            diversity_seeking_irs = self._compute_diversity_seeking_intrinsic_rewards(
                    obs_embeddings=obs_embeddings,
                    next_obs_embeddings=next_obs_embeddings)
            all_irs += self.diversity_seeking_ir_weight * diversity_seeking_irs

        if self.residual_error_ir_weight is not None:
            residual_error_irs = self._compute_residual_error_intrinsic_rewards(
                    residual_errors_before=residual_errors_before,
                    residual_errors_after=residual_errors_after,
                    unnormalized_residual_errors_before=unnormalized_residual_errors_before,
                    unnormalized_residual_errors_after=unnormalized_residual_errors_after)
            all_irs += self.residual_error_ir_weight * residual_error_irs

        logger.record_tabular('AllIntrinsicRewardsMax', np.max(all_irs))
        logger.record_tabular('AllIntrinsicRewardsMin', np.min(all_irs))
        logger.record_tabular('AllIntrinsicRewardsMean', np.mean(all_irs))

        return all_irs

    def _compute_residual_error_intrinsic_rewards(self,
                                                  *,
                                                  residual_errors_before,
                                                  residual_errors_after,
                                                  unnormalized_residual_errors_before,
                                                  unnormalized_residual_errors_after):
        if self.residual_error_ir_calc_after_opt:
            if self.residual_error_ir_use_unnormalized_errors:
                residual_errors = unnormalized_residual_errors_after
            else:
                residual_errors = residual_errors_after
        else:
            if self.residual_error_ir_use_unnormalized_errors:
                residual_errors = unnormalized_residual_errors_before
            else:
                residual_errors = residual_errors_before

        assert residual_errors is not None

        logger.record_tabular('ResidualMax', np.max(residual_errors))
        logger.record_tabular('ResidualMin', np.min(residual_errors))
        logger.record_tabular('ResidualMean', np.mean(residual_errors))

        if self.residual_error_ir_normalize:
            residual_errors = scale_values(residual_errors)

        return residual_errors

    def _compute_diversity_seeking_intrinsic_rewards(self,
                                                     *,
                                                     obs_embeddings,
                                                     next_obs_embeddings):
        if self.diversity_seeking_pool == 'replay_pool':
            ordered = None
            pool_target = self.model.eval_state_embeddings(self.replay_pool.get_data(index=1, ordered=ordered))
        elif self.diversity_seeking_pool == 'batch':
            pool_target = obs_embeddings
        else:
            assert False

        next_target = next_obs_embeddings
        current_target = obs_embeddings

        similarities_next_obses = self._compute_similarities(
                target=next_target,
                kernel_bandwidth=self.diversity_seeking_kernel_bandwidth,
                pool_target=pool_target)
        similarities_final = similarities_next_obses

        logger.record_tabular('NaiveDiversityMax', -np.min(similarities_next_obses))
        logger.record_tabular('NaiveDiversityMin', -np.max(similarities_next_obses))
        logger.record_tabular('NaiveDiversityMean', -np.mean(similarities_next_obses))

        if self.diversity_seeking_calc == 'relative':
            similarities_obses = self._compute_similarities(
                    target=current_target,
                    kernel_bandwidth=self.diversity_seeking_kernel_bandwidth,
                    pool_target=pool_target)
            similarities_final = similarities_next_obses - similarities_obses

            logger.record_tabular('RelativeDiversityMax', -np.min(similarities_final))
            logger.record_tabular('RelativeDiversityMin', -np.max(similarities_final))
            logger.record_tabular('RelativeDiversityMean', -np.mean(similarities_final))
        elif self.diversity_seeking_calc == 'naive':
            pass
        else:
            assert False

        diversities = -similarities_final
        return diversities

    def _compute_similarities(self,
                              *,
                              target,
                              kernel_bandwidth,
                              pool_target):
        division_factor = (2.0 * np.square(kernel_bandwidth))

        if self.compute_similarities_on_gpu:
            logger.log('Computing similarities on GPU')
            return self.similarity_calculator.compute_similarities(
                    target, pool_target, division_factor)

        norm_squares = np.sum(np.square(target[:, np.newaxis, :] - pool_target), axis=-1)
        return np.mean(np.exp(-norm_squares / division_factor), axis=1)

    def get_itr_snapshot(self, itr, samples_data):
        result = dict(
                dsae_embedding=self.model)
        return result

