import unittest
from unittest.mock import MagicMock

import strategies as strat
import neal
import samplers
import circuits as c

datasource_1 = [
    {"x_data": [0, 1], "y_data": 1, "solution": {'s_0': 0}},
    {"x_data": [0, 1], "y_data": 0, "solution": {'s_0': 1}},
    {"x_data": [1, 1], "y_data": 1, "solution": {'s_0': 1}},
    {"x_data": [1, 1, 0], "y_data": 1, "solution": {'s_0': 0, 's_1': 0, 's_2': 1, 'z_0': 'irrelevant'}},
    {"x_data": [1, 1, 0], "y_data": 1, "solution": {'s_0': 0, 's_1': 1, 's_2': 0, 'z_0': 'irrelevant'}},
]

datasource_2 = [
    {"x_data": [0, 1], "y_data": 1, "solution": {'s_0': 1}},
    {"x_data": [0, 1], "y_data": 0, "solution": {'s_0': 0}},
    {"x_data": [1, 1, 0], "y_data": 0, "solution": {'s_0': 0, 's_1': 1, 's_2': 0, 'z_0': 'irrelevant'}},
    {"x_data": [1, 0, 1], "y_data": 1, "solution": {'s_0': 1, 's_1': 1, 's_2': 1, 'z_0': 'irrelevant'}}
]

class MockClass(dict):
    __getattr__, __setattr__ = dict.get, dict.__setitem__

def make_result(return_vals):
    new_return_vals = []
    for v in return_vals:
        val = MockClass()
        val.sample = v
        new_return_vals += [val]

    result = MockClass()
    result.data = lambda x: new_return_vals
    result.info = {}
    return result


class CheckStrategies(unittest.TestCase):

    def test_check_solution(self):
        """ Should be able to correctly identify a solution """

        strategy = strat.BatchStrategy(1, 100, neal.SimulatedAnnealingSampler(), 1)

        for test in datasource_1:
            x_data = test["x_data"]
            y_data = test["y_data"]
            solution = test["solution"]

            result = strategy.check_solution_for_row(x_data, y_data, solution)

            self.assertTrue(result)

    def test_check_solution_negative(self):
        """ Should be able to correctly identify an incorrect solution """

        strategy = strat.BatchStrategy(1, 100, neal.SimulatedAnnealingSampler(), 1)

        for test in datasource_2:
            x_data = test["x_data"]
            y_data = test["y_data"]
            solution = test["solution"]

            result = strategy.check_solution_for_row(x_data, y_data, solution)

            self.assertFalse(result)

    def test_solve_batch(self):
        """ should select solutions from the sampler which are actually correct. """

        #  Given
        solutions = [make_result([{'s_0': 1}, {'s_0': 0}])]  # one real one, one fake one
        x_rows = [[1, 0], [1, 1]]
        y_rows = [0, 1]

        expected = { (('s_0', 1),) }

        sampler = samplers.MockSampler(solutions)
        strategy = strat.BatchStrategy(1, 100, sampler, 1)

        # When
        result = strategy.solve_batch(x_rows, y_rows)

        # Then
        self.assertEqual(result, expected)

    def test_solve(self):
        """ solve method should take the intersection over all results for the individual subproblems. """

        # Given
        all_ones = (('s_0', 1), ('s_1', 1), ('s_2', 1),)
        results = [
            {
                all_ones,
                (('s_0', 0), ('s_1', 0), ('s_2', 0),),
            },
            {
                all_ones,
                (('s_0', 1), ('s_1', 0), ('s_2', 0),),
            },
            {
                all_ones,
                (('s_0', 0), ('s_1', 0), ('s_2', 1),),
            },
            {
                all_ones,
                (('s_0', 0), ('s_1', 1), ('s_2', 0),),
            },
        ]

        sampler = samplers.MockSampler([])
        strategy = strat.BatchStrategy(2, 100, sampler, 4)
        mock_solve_batch = MagicMock()
        mock_solve_batch.side_effect = results
        strategy.solve_batch = mock_solve_batch

        expected = [strategy.convert_tuples_to_dict(all_ones)]

        # When
        result = strategy.solve([], [])

        # Then
        self.assertEqual(result, expected)

    def test_solve_2(self):
        """ it should call solve_batch with all x and y combinations"""

        # Given
        weights = [1, 0, 1]
        n_layers = 2
        sampler = samplers.MockSampler([])
        strategy = strat.BatchStrategy(n_layers, 100, sampler, 4)
        mock_solve_batch = MagicMock(return_value={(('s_0', 1), ('s_1', 1), ('s_2', 1),)})
        strategy.solve_batch = mock_solve_batch

        actual_circuit = c.make_specific_circuit(weights)
        x_data, y_data = c.make_complete_data(actual_circuit, n_layers)

        # When
        strategy.solve(x_data, y_data)
        calls = mock_solve_batch.call_args_list
        x_calls = [ c for call in calls for c in call[0][0]]
        y_calls = [ c for call in calls for c in call[0][1]]
        for x_row, y_row in zip(x_data, y_data):
            self.assertTrue(x_row in x_calls)
            self.assertTrue(y_row in y_calls)

    def test_solve_2_1(self):
        """ it should call solve_batch with all x and y combinations, when n_batches is 1"""

        # Given
        weights = [1, 0, 1, 0, 1, 0]
        n_layers = 3
        sampler = samplers.MockSampler([])
        strategy = strat.BatchStrategy(n_layers, 100, sampler, 1)
        mock_solve_batch = MagicMock(return_value={(('s_0', 1), ('s_1', 1), ('s_2', 1),)})
        strategy.solve_batch = mock_solve_batch

        actual_circuit = c.make_specific_circuit(weights)
        x_data, y_data = c.make_complete_data(actual_circuit, n_layers)

        # When
        strategy.solve(x_data, y_data)
        calls = mock_solve_batch.call_args_list
        x_calls = [ c for call in calls for c in call[0][0]]
        y_calls = [ c for call in calls for c in call[0][1]]
        for x_row, y_row in zip(x_data, y_data):
            self.assertTrue(x_row in x_calls)
            self.assertTrue(y_row in y_calls)

    def test_solve_2_2(self):
        """ it should call solve_batch with all x and y combinations, when n_batches is size of dataset"""

        # Given
        weights = [1, 0, 1, 0, 1, 0]
        n_layers = 3
        sampler = samplers.MockSampler([])
        strategy = strat.BatchStrategy(n_layers, 100, sampler, 2 ** 4)
        mock_solve_batch = MagicMock(return_value={(('s_0', 1), ('s_1', 1), ('s_2', 1),)})
        strategy.solve_batch = mock_solve_batch

        actual_circuit = c.make_specific_circuit(weights)
        x_data, y_data = c.make_complete_data(actual_circuit, n_layers)

        # When
        strategy.solve(x_data, y_data)
        calls = mock_solve_batch.call_args_list
        x_calls = [ c for call in calls for c in call[0][0]]
        y_calls = [ c for call in calls for c in call[0][1]]
        for x_row, y_row in zip(x_data, y_data):
            self.assertTrue(x_row in x_calls)
            self.assertTrue(y_row in y_calls)

