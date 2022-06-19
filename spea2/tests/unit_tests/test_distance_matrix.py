"""Unit test meal_plan_algorithm.spea2.distance_matrix."""
import numpy as np

import unittest
import numpy.testing as npt
from hypothesis import given, settings
from tests.unit_tests.hypothesis_strategies import rnd_nd_dims

test_data_location = "tests/unit_tests/testdata_distance_matrix/"

import spea2.distance_matrix as distance_matrix


class TestDistanceMatrix(unittest.TestCase):
    @settings(max_examples=5)
    @given(rnd_nd_dims(2, 1, 50))
    def test_calculate_zeros(self, dims):
        """Assert the function produces a 0 matrix for 0 array inputs."""
        inputdata = np.zeros(dims)
        columns = dims[0]
        expected_output = np.zeros((columns, columns))
        output = distance_matrix.calculate(inputdata)
        npt.assert_allclose(output, expected_output)

    def test_calculate_sample_data(self):
        """Assert the function produces correct output for a sample input set."""
        inputdata = np.loadtxt(test_data_location + "input.ndarr")
        output = np.loadtxt(test_data_location + "output.ndarr")
        npt.assert_allclose(distance_matrix.calculate(inputdata), output)


if __name__ == "__main__":
    unittest.main()
