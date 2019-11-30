import unittest
import strategies as strat
import neal


datasource_1 = [
    { "x_data": [0, 1], "y_data": 1, "solution": { 's_0': 0 } },
    { "x_data": [0, 1], "y_data": 0, "solution": { 's_0': 1 } },
    { "x_data": [1, 1], "y_data": 1, "solution": { 's_0': 1 } },
    { "x_data": [1, 1, 0], "y_data": 1, "solution": { 's_0': 0, 's_1': 0, 's_2': 1, 'z_0': 'irrelevant' } },
    { "x_data": [1, 1, 0], "y_data": 1, "solution": { 's_0': 0, 's_1': 1, 's_2': 0, 'z_0': 'irrelevant' } },
]

datasource_2 = [
    { "x_data": [0, 1], "y_data": 1, "solution": { 's_0': 1 } },
    { "x_data": [0, 1], "y_data": 0, "solution": { 's_0': 0 } },
    { "x_data": [1, 1, 0], "y_data": 0, "solution": { 's_0': 0, 's_1': 1, 's_2': 0, 'z_0': 'irrelevant' } },
    { "x_data": [1, 0, 1], "y_data": 1, "solution": { 's_0': 1, 's_1': 1, 's_2': 1, 'z_0': 'irrelevant' } }
]

class CheckCircuits(unittest.TestCase):

    def test_check_solution(self):
        """ Should be able to correctly identify a solution """

        strategy = strat.EliminationStrategy(1, [], neal.SimulatedAnnealingSampler())

        for test in datasource_1:
            x_data = test["x_data"]
            y_data = test["y_data"]
            solution = test["solution"]

            result = strategy.check_solution(x_data, y_data, solution)

            self.assertTrue(result)

    def test_check_solution_negative(self):
        """ Should be able to correctly identify an incorrect solution """

        strategy = strat.EliminationStrategy(1, [], neal.SimulatedAnnealingSampler())

        for test in datasource_2:
            x_data = test["x_data"]
            y_data = test["y_data"]
            solution = test["solution"]

            result = strategy.check_solution(x_data, y_data, solution)

            self.assertFalse(result)
