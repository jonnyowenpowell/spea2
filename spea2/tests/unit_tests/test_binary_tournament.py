"""Unit test meal_plan_algorithm.spea2.binary_tournament."""
import numpy as np

import unittest
from hypothesis import given, settings

from tests.unit_tests.hypothesis_strategies import rnd_len_1d_arrays

import spea2.binary_tournament as binary_tournament


class TestBinaryTournament(unittest.TestCase):
    @settings(max_examples=10)
    @given(rnd_len_1d_arrays(np.float64, 2, 200))
    def test_perform(self, fitnesses):
        """Assert the function selects an index in the population."""
        output = binary_tournament.perform(fitnesses)
        self.assertTrue(type(output) is int)
        self.assertTrue(0 <= output and output < len(fitnesses))


if __name__ == "__main__":
    unittest.main()
