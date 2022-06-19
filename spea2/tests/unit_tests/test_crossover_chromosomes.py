"""Unit test meal_plan_algorithm.spea2.crossover_chromosomes."""
import numpy as np

import unittest
from hypothesis import given, settings, assume
from hypothesis.strategies import floats

from tests.unit_tests.hypothesis_strategies import rnd_len_1d_array_pairs

import spea2.crossover_chromosomes as crossover_chromosomes


class TestCrossoverChromosomes(unittest.TestCase):
    @settings(max_examples=5)
    @given(
        rnd_len_1d_array_pairs(np.float64, 1, 50),
        floats(min_value=0.01, max_value=0.499),
    )
    def test_perform_linear_combination(self, parents, gamma):
        """Assert the function produces linear combinatnations of the two parents with gamma in the specified range."""
        parent_a, parent_b = parents
        length = parent_a.shape[0]
        crossover_denominators = parent_a - parent_b
        crossover_non_zero_bool = np.not_equal(crossover_denominators, 0)
        assume(np.sum(crossover_non_zero_bool) > 0)

        child_a, child_b = crossover_chromosomes.perform(
            parent_a, parent_b, length, gamma
        )

        crossover_numerators = child_a - parent_b
        crossover_factors = (
            crossover_numerators[crossover_non_zero_bool]
            / crossover_denominators[crossover_non_zero_bool]
        )
        self.assertTrue(np.greater_equal(np.amin(crossover_factors), -1.0 * gamma))
        self.assertTrue(np.less_equal(np.amax(crossover_factors), 1.0 + gamma))

    @settings(max_examples=5)
    @given(
        rnd_len_1d_array_pairs(np.float64, 1, 50),
        floats(min_value=0.01, max_value=0.499),
    )
    def test_perform_output_type(self, parents, gamma):
        """Assert the function produces a pair of chromosomes with the same length and dtype as the parents."""
        parent_a, parent_b = parents
        length = parent_a.shape[0]
        child_a, child_b = crossover_chromosomes.perform(
            parent_a, parent_b, length, gamma
        )
        self.assertEqual(length, child_a.shape[0])
        self.assertEqual(length, child_b.shape[0])
        self.assertEqual(parent_a.dtype, child_a.dtype)
        self.assertEqual(parent_a.dtype, child_b.dtype)


if __name__ == "__main__":
    unittest.main()
