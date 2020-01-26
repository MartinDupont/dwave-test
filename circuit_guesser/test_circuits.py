import unittest
import itertools
import circuits as c

def sort_tuple_pairs(input_tuple):
    return tuple(sorted(t for t in input_tuple))

def sort_tuple_list(tuple_list):
    return sorted([sort_tuple_pairs(t) for t in tuple_list])


class CheckCircuits(unittest.TestCase):

    def test_control_gate(self):
        """ When s=1 it should deliver a and when s=0 it should deliver b"""
        a_list = [True, False, True, False, True, False, True, False]
        b_list = [True, True, False, False, True, True, False, False]
        s_list = [True, True, True, True, False, False, False, False]
        expected = [True, False, True, False, True, True, False, False]

        results = [c.control_gate(s, a, b) for s, a, b in zip(s_list, a_list, b_list)]
        self.assertEqual(results, expected)

    def test_recursive_circuit_n_1(self):
        """ Should deliver the base case, which delivers an AND gate for s=1 and an OR gate for s=0 """
        s_list = [True, True, True, True, False, False, False, False]
        a_list = [True, False, True, False, True, False, True, False]
        b_list = [True, True, False, False, True, True, False, False]
        expected = [True, False, False, False, True, True, True, False]

        x_inputs = zip(a_list, b_list)
        results = [c.recursive_circuit([s], x) for s, x in zip(s_list, x_inputs)]
        self.assertEqual(results, expected)

    def test_recursive_circuit_n_2(self):
        """ Should deliver the tree with 3 s parameters and 3 x parameters """

        # Given
        def reference_function(s, x):
            return c.base_gate(s[2], c.base_gate(s[0], x[0], x[1]), c.base_gate(s[1], x[1], x[2]))

        all_combinations = list(itertools.product([False, True], repeat=6))
        x_values = [(c[0], c[1], c[2]) for c in all_combinations]
        s_values = [(c[3], c[4], c[5]) for c in all_combinations]
        expected = [reference_function(s, x) for s, x in zip(s_values, x_values)]

        # When
        results = [c.recursive_circuit(s, x) for s, x in zip(s_values, x_values)]

        # Then
        for x, s, e, r in zip(x_values, s_values, expected, results):
            self.assertEqual(r, e)

    def test_wrap_with_complete_data_n_1(self):
        """ given some initial weights, it should give back a function which returns true when given the initial weights"""
        all_combinations = [[False], [True]]

        for s in all_combinations:
            specific_circuit = c.make_specific_circuit(s)
            wrapped_circuit = c.wrap_with_complete_data(specific_circuit, 1)
            self.assertTrue(wrapped_circuit(s))

    def test_wrap_with_complete_data_n_2(self):
        """ given some initial weights, it should give back a function which returns true when given the initial weights"""
        all_combinations = list(itertools.product([False, True], repeat=3))

        for s in all_combinations:
            specific_circuit = c.make_specific_circuit(s)
            wrapped_circuit = c.wrap_with_complete_data(specific_circuit, 2)
            self.assertTrue(wrapped_circuit(s))

    def test_wrap_with_complete_data_n_3(self):
        """ given some initial weights, it should give back a function which returns true when given the initial weights"""
        all_combinations = list(itertools.product([False, True], repeat=6))

        for s in all_combinations:
            specific_circuit = c.make_specific_circuit(s)
            wrapped_circuit = c.wrap_with_complete_data(specific_circuit, 3)
            self.assertTrue(wrapped_circuit(s))

    def test_make_base_polynomial(self):
        """ given variable names, it should output a dictionary """
        y = "y"
        z_1 = "z_1"
        z_2 = "z_2"
        s = "s"
        result = c.make_base_polynomial(y, z_1, z_2, s)

        expected = {("y",): 1, ("z_1",): 1, ("z_2",): 1, ("y", "z_1"): -2, ("y", "z_2"): -2, ("y", "s"): 2,
                    ("z_1", "s"): -1, ("z_2", "s"): -1, ("z_1", "z_2"): 1}
        self.assertEqual(result, expected)

    def test_make_output_polynomial(self):
        """ given variable names, it should output a dictionary """
        z_1 = "z_1"
        z_2 = "z_2"
        s = "s"

        result_0 = c.make_output_polynomial(0, z_1, z_2, s)
        expected_0 = {("z_1",): 1, ("z_2",): 1, ("z_1", "s"): -1, ("z_2", "s"): -1, ("z_1", "z_2"): 1}
        self.assertEqual(result_0, expected_0)

        result_1 = c.make_output_polynomial(1, z_1, z_2, s)
        expected_1 = {("z_1",): -1, ("z_2",): -1, ("s",): 2, ("z_1", "s"): -1, ("z_2", "s"): -1, ("z_1", "z_2"): 1}
        self.assertEqual(result_1, expected_1)

    def test_make_input_polynomial(self):
        """ given variable names, it should output a dictionary """
        y = "y"
        s = "s"

        result_00 = c.make_input_polynomial(y, 0, 0, s)
        expected_00 = {("y",): 1, ("y", "s"): 2}
        self.assertEqual(result_00, expected_00)

        result_01 = c.make_input_polynomial(y, 0, 1, s)
        expected_01 = {("y",): -1, ("y", "s"): 2, ("s",): -1}
        self.assertEqual(result_01, expected_01)

        result_10 = c.make_input_polynomial(y, 1, 0, s)
        expected_10 = {("y",): -1, ("y", "s"): 2, ("s",): -1}
        self.assertEqual(result_10, expected_10)

        result_11 = c.make_input_polynomial(y, 1, 1, s)
        expected_11 = {("y",): -3, ("y", "s"): 2, ("s",): -2}
        self.assertEqual(result_11, expected_11)

    def test_merge_dicts(self):
        dict1 = {"a": 1, "b": 1}
        dict2 = {"b": 1, "c": 1}
        expected = {"a": 1, "b": 2, "c": 1}

        result = c.merge_dicts_and_add(dict1, dict2)

        self.assertEqual(expected, result)

    def test_merge_dicts2(self):
        dict1 = {"a": 1, "b": 1}
        dict2 = {"b": 1, "c": 1}
        dict3 = {"c": 1, "d": 1}
        expected = {"a": 1, "b": 2, "c": 2, "d": 1}

        result = c.merge_dicts_and_add(dict1, dict2, dict3)

        self.assertEqual(expected, result)

    def test_make_polynomial_for_datapoint_trivial(self):
        x_vals = [0, 0]
        y = 0

        polynomial, _ = c.make_polynomial_for_datapoint(y, x_vals)

        expected = {('s_0',) : 0}

        self.assertEqual(polynomial, expected)

    def test_make_polynomial_for_datapoint(self):
        x_vals = [0, 0, 0]
        y = 0

        polynomial, _ = c.make_polynomial_for_datapoint(y, x_vals)

        first_layer = {
            ("z_0",): 1, ("z_0", "s_0"): 2,
            ("z_1",): 1, ("z_1", "s_1"): 2}

        second_layer = {
            ("z_0",): 1, ("z_1",): 1, ("z_0", "s_2"): -1, ("z_1", "s_2"): -1, ("z_0", "z_1"): 1  # end last layer
        }
        expected = c.merge_dicts_and_add(first_layer, second_layer)

        self.assertEqual(polynomial, expected)

    def test_make_polynomial_for_datapoint_again(self):
        x_vals = [1, 1, 1]
        y = 1

        polynomial, _ = c.make_polynomial_for_datapoint(y, x_vals)

        first_layer = {
            ("z_0",): -3, ("z_0", "s_0"): 2, ("s_0",): -2,
            ("z_1",): -3, ("z_1", "s_1"): 2, ("s_1",): -2
            }

        second_layer = {("z_0",): -1, ("z_1",): -1, ("s_2",): 2, ("z_0", "s_2"): -1, ("z_1", "s_2"): -1, ("z_0", "z_1"): 1}
        expected = c.merge_dicts_and_add(first_layer, second_layer)

        self.assertEqual(polynomial, expected)

    def test_make_polynomial_for_datapoint_3_layers(self):
        x_vals = [0, 0, 0, 0]
        y = 0

        polynomial, _ = c.make_polynomial_for_datapoint(y, x_vals)

        first_layer = {
            ("z_0",): 1, ("z_0", "s_0"): 2,
            ("z_1",): 1, ("z_1", "s_1"): 2,
            ("z_2",): 1, ("z_2", "s_2"): 2}
        second_layer = [{
            ("z_3",): 1, ("z_0",): 1, ("z_1",): 1,
            ("z_3", "z_0"): -2, ("z_3", "z_1"): -2, ("z_3", "s_3"): 2,
            ("z_0", "s_3"): -1, ("z_1", "s_3"): -1, ("z_0", "z_1"): 1},  # end first block of second layer
            {("z_4",): 1, ("z_1",): 1, ("z_2",): 1,
             ("z_4", "z_1"): -2, ("z_4", "z_2"): -2, ("z_4", "s_4"): 2,
             ("z_1", "s_4"): -1, ("z_2", "s_4"): -1, ("z_1", "z_2"): 1}  # end second block of second layer
        ]

        third_layer = {
            ("z_3",): 1, ("z_4",): 1, ("z_3", "s_5"): -1, ("z_4", "s_5"): -1, ("z_3", "z_4"): 1  # end last layer
        }
        expected = c.merge_dicts_and_add(first_layer, *second_layer, third_layer)

        self.assertEqual(polynomial, expected)

    def test_make_polynomial_for_datapoint_3_layers_again(self):
        x_vals = [1, 1, 1, 1]
        y = 1

        polynomial, _ = c.make_polynomial_for_datapoint(y, x_vals)

        first_layer = {
            ("z_0",): -3, ("z_0", "s_0"): 2, ("s_0",): -2,
            ("z_1",): -3, ("z_1", "s_1"): 2, ("s_1",): -2,
            ("z_2",): -3, ("z_2", "s_2"): 2, ("s_2",): -2
        }

        second_layer = [{
            ("z_3",): 1, ("z_0",): 1, ("z_1",): 1,
            ("z_3", "z_0"): -2, ("z_3", "z_1"): -2, ("z_3", "s_3"): 2,
            ("z_0", "s_3"): -1, ("z_1", "s_3"): -1, ("z_0", "z_1"): 1},  # end first block of second layer
            {("z_4",): 1, ("z_1",): 1, ("z_2",): 1,
             ("z_4", "z_1"): -2, ("z_4", "z_2"): -2, ("z_4", "s_4"): 2,
             ("z_1", "s_4"): -1, ("z_2", "s_4"): -1, ("z_1", "z_2"): 1}  # end second block of second layer
        ]

        third_layer = {
            ("z_3",): -1, ("z_4",): -1, ("s_5",): 2, ("z_3", "s_5"): -1, ("z_4", "s_5"): -1, ("z_3", "z_4"): 1
        }
        expected = c.merge_dicts_and_add(first_layer, *second_layer, third_layer)

        self.assertEqual(polynomial, expected)

    def test_make_polynomial_for_datapoint_4_layers_rough(self):
        x_vals = [1, 1, 1, 1, 1]
        y = 1

        polynomial, _ = c.make_polynomial_for_datapoint(y, x_vals)


        expected_keys = [
            ("z_0",),
            ("z_1",),
            ("z_2",),
            ("z_3",),
            ("z_4",),
            ("z_5",),
            ("z_6",),
            ("z_7",),
            ("z_8",),
            ("s_0",),
            ("s_1",),
            ("s_2",),
            ("s_3",),
            ("s_9",),
            ("z_0", "z_1"),
            ("z_0", "z_4"),
            ("z_1", "z_4"),
            ("z_1", "z_2"),
            ("z_1", "z_5"),
            ("z_2", "z_5"),
            ("z_2", "z_3"),
            ("z_2", "z_6"),
            ("z_3", "z_6"),
            ("z_4", "z_5"),
            ("z_4", "z_7"),
            ("z_5", "z_7"),
            ("z_5", "z_6"),
            ("z_5", "z_8"),
            ("z_6", "z_8"),
            ("z_7", "z_8"),
            ("z_0", "s_0"),
            ("z_1", "s_1"),
            ("z_2", "s_2"),
            ("z_3", "s_3"),
            ("z_4", "s_4"),
            ("z_5", "s_5"),
            ("z_6", "s_6"),
            ("z_7", "s_7"),
            ("z_8", "s_8"),
            ("z_0", "s_4"),
            ("z_1", "s_4"),
            ("z_1", "s_5"),
            ("z_2", "s_5"),
            ("z_2", "s_6"),
            ("z_3", "s_6"),
            ("z_4", "s_7"),
            ("z_5", "s_7"),
            ("z_5", "s_8"),
            ("z_6", "s_8"),
            ("z_7", "s_9"),
            ("z_8", "s_9"),
        ]

        self.assertEqual(sort_tuple_list(polynomial.keys()), sort_tuple_list(expected_keys))


    def test_make_polynomial_with_start(self):
        x_vals = [1, 1, 1, 1]
        y = 1

        start = 1
        expected_count = 6
        expected_zs = ["z_1", "z_2", "z_3", "z_4", "z_5"]


        polynomial, final_z_count = c.make_polynomial_for_datapoint(y, x_vals, start)

        actual_zs = []
        for key in polynomial.keys():
            for k in key:
                if k[0] == "z":
                    actual_zs += [k]

        actual_zs = sorted(list(set(actual_zs)))


        self.assertEqual(actual_zs, expected_zs)
        self.assertEqual(expected_count, final_z_count)

    def test_make_polynomials_for_many_datapoints(self):

        x_vals = [[0, 0, 0], [1, 1, 1]]
        y_vals = [0, 1]

        polynomial = c.make_polynomial_for_many_datapoints(y_vals, x_vals)

        first_layer = {
            ("z_0",): 1, ("z_0", "s_0"): 2,
            ("z_1",): 1, ("z_1", "s_1"): 2,
            ("z_2",): -3, ("z_2", "s_0"): 2, ("s_0",): -2,
            ("z_3",): -3, ("z_3", "s_1"): 2, ("s_1",): -2
            }

        second_layer = {
            ("z_0",): 1, ("z_1",): 1, ("z_0", "s_2"): -1, ("z_1", "s_2"): -1, ("z_0", "z_1"): 1,
            ("z_2",): -1, ("z_3",): -1, ("s_2",): 2, ("z_2", "s_2"): -1, ("z_3", "s_2"): -1, ("z_2", "z_3"): 1
        }
        expected = c.merge_dicts_and_add(first_layer, second_layer)

        self.assertEqual(polynomial, expected)


if __name__ == "__main__":
    unittest.main()
