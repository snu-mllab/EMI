import numpy as np
import tensorflow as tf

import rllab.misc.logger as logger

from dsae.utils import tf_print

def compute_sample_covariance(centered_data, sample_size, name):
    covariance = tf.matmul(1.0 / (sample_size-1.0) * centered_data,
                           centered_data, transpose_a=True, transpose_b=False)

    # Consider case of zero covariance
    almost_zero_covariance = tf.fill(tf.shape(covariance), 1e-10)
    abs_sum = tf.reduce_sum(tf.abs(covariance))
    cond = tf.equal(abs_sum, 0)
    covariance = tf.where(cond, almost_zero_covariance, covariance)

    covariance = tf_print(covariance, ['compute_sample_covariance', name, covariance])
    return covariance

def compute_mahalanobis_distance_impl(errors, covariance, sample_size, fudge=1e-6):
    covariance_orig = covariance
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    covariance = tf_print(covariance, ['compute_mahalanobis_distance_impl', 'given covariance: ', covariance_orig, 'symmetric covariance: ', covariance])
    ee, vv = tf.self_adjoint_eig(covariance)
    ee = tf_print(ee, ['compute_mahalanobis_distance_impl', 'eigenvalues: ', ee, 'condition number: ', tf.reduce_max(ee) / tf.reduce_min(ee)])
    ee = tf.maximum(ee, fudge)
    ee_inv = tf.divide(1.0, ee)
    ee_inv_sqrt = tf.sqrt(ee_inv)
    matrix_half = tf.matmul(errors, vv, transpose_a=False, transpose_b=False)
    matrix_half = tf.matmul(matrix_half, tf.diag(ee_inv_sqrt),
                            transpose_a=False, transpose_b=False)
    return tf.reduce_sum(tf.square(matrix_half), axis=1)

def compute_mahalanobis_distance(embeddings, errors, name, fudge=1e-6):
    size = tf.shape(embeddings)[0]
    embeddings_mean = tf.reduce_mean(embeddings, 0)
    embeddings_shifted = embeddings - embeddings_mean
    embeddings_covariance = compute_sample_covariance(
            embeddings_shifted, size, name + '_covariance')
    mahalanobis_dists = compute_mahalanobis_distance_impl(
            errors=errors, covariance=embeddings_covariance, sample_size=size, fudge=fudge)
    return mahalanobis_dists

def multivariate_kl_with_unit_gaussian_sym(mean, covariance, embedding_dimension, fudge=1e-6):
    covariance_orig = covariance
    covariance = 0.5 * (covariance + tf.transpose(covariance))
    covariance = tf_print(covariance, ['multivariate_kl_with_unit_gaussian_sym', 'given covariance: ', covariance_orig, 'symmetric covariance: ', covariance])

    ee, _ = tf.self_adjoint_eig(covariance)
    ee = tf_print(ee, ['multivariate_kl_with_unit_gaussian_sym', 'eigenvalues: ', ee, 'condition number: ', tf.reduce_max(ee) / tf.reduce_min(ee)])
    ee = tf.maximum(ee, fudge)

    trace = tf.reduce_sum(ee)
    log_det = tf.reduce_sum(tf.log(ee))

    result = 0.5 * (trace + (tf.reduce_sum(tf.square(mean)) if mean is not None else 0.0) - embedding_dimension - log_det)
    result = tf_print(result, ['multivariate_kl_with_unit_gaussian_sym', 'log det: ', log_det, 'result: ', result])
    return result
