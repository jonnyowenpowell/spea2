"""Unit test meal_plan_algorithm.spea2.dominated_scores."""
import numpy as np

import unittest
import numpy.testing as npt

test_data_location = "tests/unit_tests/testdata_dominated_scores/"

import spea2.domination_scores as domination_scores


class TestDominatedScores(unittest.TestCase):
    def test_calculate_sample_data(self):
        """Assert the function produces correct output for a sample input set."""
        input = np.loadtxt(test_data_location + "input.ndarr")
        output = np.loadtxt(test_data_location + "output.ndarr")
        npt.assert_allclose(domination_scores.calculate(input), output)


if __name__ == "__main__":
    unittest.main()
