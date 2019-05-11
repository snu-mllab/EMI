import tensorflow as tf

from sandbox.rocky.tf.misc import tensor_utils

from dsae.utils import tf_print, Noop

class SimilarityCalculator(object):
    def __init__(self, chunk_size=None, use_new_graph=False, use_new_session=False):
        if use_new_graph:
            self._graph = tf.Graph()
        else:
            self._graph = tf.get_default_session().graph
        self._use_new_session = use_new_session
        self._init_graph(chunk_size)

    def _init_graph(self, chunk_size):
        with self._graph.as_default():
            with tf.variable_scope('SimilarityCalculator'):
                X = tensor_utils.new_tensor(
                    'X',
                    ndim=2,
                    dtype=tf.float32,
                )
                pool = tensor_utils.new_tensor(
                    'pool',
                    ndim=2,
                    dtype=tf.float32,
                )
                division_factor = tensor_utils.new_tensor(
                    'division_factor',
                    ndim=0,
                    dtype=tf.float32,
                )

                inputs = [X, pool, division_factor]

                size = tf.shape(X)[0]

                if chunk_size is None:
                    chunk_size = size
                    chunk_size_float = tf.cast(chunk_size, tf.float32)
                else:
                    chunk_size_float = float(chunk_size)
                array_size = tf.cast(tf.ceil(tf.cast(size, tf.float32) / chunk_size_float), tf.int32)
                ta_initial = tf.TensorArray(
                        dtype=tf.float32,
                        size=array_size,
                        infer_shape=False)
                def _cond(idx, i, ta):
                    return i < size
                def _body(idx, i, ta):
                    until = tf.minimum(i + chunk_size, size)
                    new_pdiffs = (X[i:until, tf.newaxis, :] - pool)
                    squared_l2 = tf.reduce_sum(tf.square(new_pdiffs), axis=-1)
                    part_similarities = tf.reduce_mean(tf.exp(-squared_l2 / division_factor), axis=1)
                    return idx + 1, until, ta.write(idx, part_similarities)
                final_idx, final_i, ta = tf.while_loop(
                        _cond,
                        _body,
                        loop_vars=[0, 0, ta_initial],
                        parallel_iterations=1)
                result = ta.concat()

                self._get_result = tensor_utils.compile_function(
                    inputs=inputs,
                    outputs=result,
                )

    def compute_similarities(self, *args):
        with self._graph.as_default():
            if self._use_new_session:
                context = tf.Session(graph=self._graph).as_default()
            else:
                context = Noop()
            with context:
                return self._get_result(*args)

