"""Unit test meal_plan_algorithm.spea2.mutate_chromosome."""
import numpy as np

import unittest
from hypothesis import given, settings, assume
from hypothesis.strategies import floats

from tests.unit_tests.hypothesis_strategies import rnd_len_1d_arrays

import spea2.mutate_chromosome as mutate_chromosome


class TestMutateChromosome(unittest.TestCase):
    @settings(max_examples=5)
    @given(rnd_len_1d_arrays(np.float64, 1, 50), floats(min_value=0.01, max_value=10.0))
    def test_perform(self, parent, sigma):
        """Assert the function produces a chromosome with the same length and dtype as the parent."""
        length = parent.shape[0]
        child = mutate_chromosome.perform(parent, length, sigma)
        self.assertEqual(length, child.shape[0])
        self.assertEqual(parent.dtype, child.dtype)


if __name__ == "__main__":
    unittest.main()
