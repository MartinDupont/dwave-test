import unittest
import itertools
from circuits import control_gate, recursive_circuit, base_gate, make_specific_circuit, wrap_with_complete_data

class CheckCircuits(unittest.TestCase):

    def test_control_gate(self):
        """ When s=1 it should deliver a and when s=0 it should deliver b"""
        a_list = [True, False, True, False, True, False, True, False]
        b_list = [True, True, False, False, True, True, False, False]
        s_list = [True, True, True, True, False, False, False, False]
        expected = [True, False, True, False, True, True, False, False]

        results = [ control_gate(s, a, b) for s, a, b in zip(s_list, a_list, b_list)]
        self.assertEqual(results, expected)

    def test_recursive_circuit_n_1(self):
        """ Should deliver the base case, which delivers an AND gate for s=1 and an OR gate for s=0 """
        s_list = [True, True, True, True, False, False, False, False]
        a_list = [True, False, True, False, True, False, True, False]
        b_list = [True, True, False, False, True, True, False, False]
        expected = [True, False, False, False, True, True, True, False]

        x_inputs = zip(a_list, b_list)
        results = [recursive_circuit([s], x) for s, x in zip(s_list, x_inputs)]
        self.assertEqual(results, expected)

    def test_recursive_circuit_n_2(self):
        """ Should deliver the tree with 3 s parameters and 3 x parameters """

        # Given
        def reference_function(s, x):
            return base_gate(s[2], base_gate(s[0], x[0], x[1]), base_gate(s[1], x[1], x[2]))

        all_combinations = list(itertools.product([False, True], repeat=6))
        x_values = [(c[0], c[1], c[2]) for c in all_combinations]
        s_values = [(c[3], c[4], c[5]) for c in all_combinations]
        expected = [reference_function(s, x) for s, x in zip(s_values, x_values)]

        # When
        results = [recursive_circuit(s, x) for s, x in zip(s_values, x_values)]

        # Then
        for x, s, e, r in zip(x_values, s_values, expected, results):
            self.assertEqual(r, e)

    def test_wrap_with_complete_data_n_1(self):
        """ given some initial weights, it should give back a function which returns true when given the initial weights"""
        all_combinations = [[False], [True]]

        for s in all_combinations:
            specific_circuit = make_specific_circuit(s)
            wrapped_circuit = wrap_with_complete_data(specific_circuit, 1)
            self.assertTrue(wrapped_circuit(s))

    def test_wrap_with_complete_data_n_2(self):
        """ given some initial weights, it should give back a function which returns true when given the initial weights"""
        all_combinations = list(itertools.product([False, True], repeat=3))

        for s in all_combinations:
            specific_circuit = make_specific_circuit(s)
            wrapped_circuit = wrap_with_complete_data(specific_circuit, 2)
            self.assertTrue(wrapped_circuit(s))

    def test_wrap_with_complete_data_n_3(self):
        """ given some initial weights, it should give back a function which returns true when given the initial weights"""
        all_combinations = list(itertools.product([False, True], repeat=6))

        for s in all_combinations:
            specific_circuit = make_specific_circuit(s)
            wrapped_circuit = wrap_with_complete_data(specific_circuit, 3)
            self.assertTrue(wrapped_circuit(s))



if __name__ == "__main__":
    unittest.main()
