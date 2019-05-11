import numpy as np

import rllab.misc.logger as logger
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.trpo import TRPO

from dsae.utils import MeasureTime

class TRPODSAE(TRPO):
    """
    Trust Region Policy Optimization with decomposable state and action embeddings (DSAE)
    """

    def __init__(
            self,
            dsae,
            **kwargs):
        super(TRPODSAE, self).__init__(**kwargs)
        self.dsae = dsae

    @overrides
    def process_samples(self, itr, paths):
        if self.dsae is not None:
            with MeasureTime('EmbeddingProcessSamplesTime'):
                self.dsae.train_and_compute_intrinsic_rewards(itr=itr, paths=paths)

        with MeasureTime('OriginalProcessSamplesTime'):
            return super(TRPODSAE, self).process_samples(itr, paths)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        data = super(TRPODSAE, self).get_itr_snapshot(itr, samples_data)
        if self.dsae is not None:
            dsae_data = self.dsae.get_itr_snapshot(itr, samples_data)
            for key in data.keys():
                assert key not in dsae_data
            data.update(dsae_data)

        return data
