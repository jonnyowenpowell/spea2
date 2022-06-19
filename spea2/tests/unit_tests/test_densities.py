"""Unit test meal_plan_algorithm.spea2.densities."""
import numpy as np
import scipy.spatial.distance

import unittest
import numpy.testing as npt
from hypothesis import given, settings
from tests.unit_tests.hypothesis_strategies import rnd_pairwise_len_1d_arrays

test_data_location = "tests/unit_tests/meal_plan_algorithm/spea2/testdata_densities/"

import spea2.densities as densities


class TestDensities(unittest.TestCase):
    @settings(max_examples=5)
    @given(rnd_pairwise_len_1d_arrays(np.float64, 3, 50))
    def test_calculate_zeros(self, condensed_distance_matrix):
        """Assert the function decreases in k."""
        distances = scipy.spatial.distance.squareform(np.abs(condensed_distance_matrix))
        ks = range(1, len(distances))
        output = np.zeros((len(ks), len(distances)))
        for k in ks:
            output[k - 1] = densities.calculate(distances, k)
        self.assertTrue(np.all(np.diff(output, axis=1) <= 0))

    def test_calculate_sample_data(self):
        """Assert the function produces correct output for a sample input set."""
        inputdata = np.loadtxt(test_data_location + "input.ndarr")
        k = 34
        output = np.loadtxt(test_data_location + "output.ndarr")
        npt.assert_allclose(densities.calculate(inputdata, k), output)


if __name__ == "__main__":
    unittest.main()
