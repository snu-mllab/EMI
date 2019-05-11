import numpy as np

import rllab.misc.logger as logger

class ReplayPoolImpl(object):
    def __init__(
            self,
            observation_space,
            action_space,
            max_size,
            rng=None,
            extra_shapes=[],
            extra_dtypes=[],
            strategy='fifo'):

        assert strategy in ['fifo', 'reservoir']

        self.max_size = max_size
        self.observations = np.zeros(
                (max_size, observation_space.flat_dim), dtype=observation_space.dtype)
        self.next_observations = np.zeros(
                (max_size, observation_space.flat_dim), dtype=observation_space.dtype)
        self.actions = np.zeros(
                (max_size, action_space.flat_dim), dtype=action_space.dtype)
        self.extras = [
            np.zeros((max_size,) + extra_shape, dtype=np.dtype(extra_dtype)) if extra_shape is not None and extra_dtype is not None else None
            for extra_shape, extra_dtype in zip(extra_shapes, extra_dtypes)
        ]

        self._rng = rng
        self.strategy = strategy

        self.reset()

        self.data = (self.next_observations, self.observations, self.actions)

    @property
    def pool_is_batch(self):
        return False


    @property
    def rng(self):
        if self._rng is None:
            return np.random
        return self._rng

    def reset(self):
        self.observations.fill(self.observations.dtype.type())
        self.next_observations.fill(self.next_observations.dtype.type())
        self.actions.fill(self.actions.dtype.type())
        for extra in self.extras:
            if extra is not None:
                extra.fill(extra.dtype.type())

        self.add_count = 0

        self.top = 0
        self.size = 0

        if self.strategy == 'reservoir':
            self.pool_indices = np.arange(self.max_size)

    def _add_samples_fifo(self, obses, next_obses, actions, extras):
        self.add_count += len(obses)

        logger.log('ReplayPoolImpl _add_samples_fifo sample size: {}'.format(len(obses)))

        obses = obses[-self.max_size:]
        next_obses = next_obses[-self.max_size:]
        actions = actions[-self.max_size:]
        extras = [e[-self.max_size:] for e in extras]

        desired_input_size = len(obses)
        part_one_size = min(desired_input_size, (self.max_size - self.top))
        part_two_size = desired_input_size - part_one_size

        part_one_store_slice = slice(self.top, self.top + part_one_size)
        self.observations[part_one_store_slice] = obses[:part_one_size]
        self.next_observations[part_one_store_slice] = next_obses[:part_one_size]
        self.actions[part_one_store_slice] = actions[:part_one_size]
        for self_extra, given_extra in zip(self.extras, extras):
            if self_extra is not None:
                self_extra[part_one_store_slice] = given_extra[:part_one_size]

        self.top += part_one_size
        assert self.top <= self.max_size
        if self.top == self.max_size:
            self.top = 0

        if part_two_size > 0:
            part_two_store_slice = slice(0, part_two_size)
            self.observations[part_two_store_slice] = obses[-part_two_size:]
            self.next_observations[part_two_store_slice] = next_obses[-part_two_size:]
            self.actions[part_two_store_slice] = actions[-part_two_size:]
            for self_extra, given_extra in zip(self.extras, extras):
                if self_extra is not None:
                    self_extra[part_two_store_slice] = given_extra[-part_two_size:]
            self.top = part_two_size

        self.size = min(self.max_size, self.size + desired_input_size)

    def _add_samples_reservoir_impl(self, obses, next_obses, actions, extras):
        assert self.size == self.max_size
        assert self.add_count >= self.max_size

        sample_size = len(obses)

        assignments = np.zeros((self.max_size,), dtype=np.int32)
        assignments.fill(-1)

        for idx in range(sample_size):
            self.add_count += 1

            assign_index = self.rng.randint(0, self.add_count)
            if assign_index < self.max_size:
                assignments[assign_index] = idx

        assign_condition = (assignments != -1)
        dest_indices = self.pool_indices[assign_condition]
        source_indices = assignments[assign_condition]

        logger.log('ReplayPoolImpl _add_samples_reservoir_impl sample size: {}, # indices to be assigned: {}'.format(sample_size, len(dest_indices)))

        self.observations[dest_indices] = obses[source_indices]
        self.next_observations[dest_indices] = next_obses[source_indices]
        self.actions[dest_indices] = actions[source_indices]
        for self_extra, given_extra in zip(self.extras, extras):
            if self_extra is not None:
                self_extra[dest_indices] = given_extra[source_indices]

    def add_samples(self, *, obses, next_obses, actions, extras):
        if self.strategy == 'fifo':
            self._add_samples_fifo(obses, next_obses, actions, extras)
        elif self.strategy == 'reservoir':
            certain_insertion_size = min(self.max_size - self.size, len(obses))

            if certain_insertion_size > 0:
                self._add_samples_fifo(
                        obses[:certain_insertion_size],
                        next_obses[:certain_insertion_size],
                        actions[:certain_insertion_size],
                        [e[:certain_insertion_size] for e in extras])
            if certain_insertion_size < len(obses):
                self._add_samples_reservoir_impl(
                        obses[certain_insertion_size:],
                        next_obses[certain_insertion_size:],
                        actions[certain_insertion_size:],
                        [e[certain_insertion_size:] for e in extras])
        else:
            assert False

    def __len__(self):
        return self.size

    def get_data(self, index=None, ordered=None):
        if ordered is None:
            ordered = (self.strategy == 'fifo')
        if ordered:
            assert self.strategy == 'fifo'

        if self.size < self.max_size:
            if index is not None:
                return self.data[index][:self.top]
            return tuple(map(lambda x: x[:self.top], self.data))

        if ordered:
            slices = [slice(self.top, None), slice(None, self.top)]
            def _finalize(target):
                return np.concatenate(
                        [target[slices[0]], target[slices[1]]],
                        axis=0)
        else:
            def _finalize(target):
                return target
        if index is not None:
            return _finalize(self.data[index])
        return tuple(map(_finalize, self.data))

    def get_extra(self, index, ordered=None):
        assert self.extras[index] is not None

        if ordered is None:
            ordered = (self.strategy == 'fifo')
        if ordered:
            assert self.strategy == 'fifo'

        if self.size < self.max_size:
            return self.extras[index][:self.top]

        if ordered:
            slices = [slice(self.top, None), slice(None, self.top)]
            def _finalize(target):
                return np.concatenate(
                        [target[slices[0]], target[slices[1]]],
                        axis=0)
        else:
            def _finalize(target):
                return target
        return _finalize(self.extras[index])

class MockReplayPool(object):
    def __init__(
            self,
            desired_max_size,
            num_extras=0,
            rng=None,
            strategy='fifo'):

        assert strategy in ['fifo', 'subsampled_batch']

        self.max_size = desired_max_size
        self.num_extras = num_extras
        self._rng = rng
        self.strategy = strategy

        self.reset()

    @property
    def pool_is_batch(self):
        return self.max_size is None

    @property
    def rng(self):
        if self._rng is None:
            return np.random
        return self._rng

    def reset(self):
        self.observations = None
        self.next_observations = None
        self.actions = None
        self.extras = [None for _ in range(self.num_extras)]

    def add_samples(self, *, obses, next_obses, actions, extras):
        num_samples = len(obses)
        if num_samples == self.max_size or self.max_size is None:
            self.observations = obses
            self.next_observations = next_obses
            self.actions = actions
            self.extras = extras
        elif self.strategy == 'fifo':
            self.observations = obses[-self.max_size:]
            self.next_observations = next_obses[-self.max_size:]
            self.actions = actions[-self.max_size:]
            self.extras = [e[-self.max_size:] for e in extras]
        elif self.strategy == 'subsampled_batch':
            indices = self.rng.choice(
                    num_samples, self.max_size, replace=False)
            self.observations = obses[indices]
            self.next_observations = next_obses[indices]
            self.actions = actions[indices]
            self.extras = [e[indices] for e in extras]

    def __len__(self):
        if self.observations is None:
            return 0
        return self.max_size

    def get_data(self, index=None, ordered=None):
        if ordered is None:
            ordered = (self.strategy == 'fifo')
        if ordered:
            assert self.strategy == 'fifo' or self.max_size is None

        data = (self.next_observations, self.observations, self.actions)

        if index is not None:
            return data[index]
        return data

    def get_extra(self, index, ordered=None):
        if ordered is None:
            ordered = (self.strategy == 'fifo')
        if ordered:
            assert self.strategy == 'fifo' or self.max_size is None

        assert self.extras[index] is not None

        return self.extras[index]

def ReplayPool(
        *,
        observation_space,
        action_space,
        min_batch_size,
        max_size=None,
        rng=None,
        extra_shapes=[],
        extra_dtypes=[],
        strategy='fifo'):
    assert strategy in ['fifo', 'subsampled_batch', 'reservoir']
    assert len(extra_shapes) == len(extra_dtypes)
    if strategy == 'subsampled_batch' or (strategy == 'fifo' and (max_size is None or max_size <= min_batch_size)):
        return MockReplayPool(
                desired_max_size=max_size,
                num_extras=len(extra_shapes),
                rng=rng,
                strategy=strategy)
    return ReplayPoolImpl(
            observation_space=observation_space,
            action_space=action_space,
            max_size=max_size,
            rng=rng,
            extra_shapes=extra_shapes,
            extra_dtypes=extra_dtypes,
            strategy=strategy)

