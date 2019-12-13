import itertools
import random
import dimod


## maybe replace with an xor gate????
def or_gate(a, b):
    return a or b


def and_gate(a, b):
    return a and b


def control_gate(s, a, b):
    """
    Returns a or b depending on the value of s.

    :param s: Control parameter, Bool
    :param a: First input, Bool
    :param b: Second input, Bool
    :return: Bool
    """

    a_side = a and s
    b_side = b and not s
    return a_side or b_side


def base_gate(s, a, b):
    return control_gate(s, and_gate(a, b), or_gate(a, b))


def trivial_fun(x):
    return x


def sanitize_inputs(x):
    """ for x an iterable"""
    return [not not x_i for x_i in x]


def recursive_circuit(s, x):
    """
    Recursive circuit
    :param s: list/iterable of control variables s (spins in dwave)
    :param x: list/iterable of inputs
    :return: single boolean
    """
    n_s = len(s)
    n_x = len(x)
    n = n_x - 1
    if int((n_x * (n_x - 1) / 2)) != n_s:
        raise ValueError("number of s arguments must be compatible with number of x arguments")

    s = sanitize_inputs(s)
    x = sanitize_inputs(x)

    # need a base case
    if n_s <= 0:
        return x[0]

    remaining_s = s[n:]
    layer_output = [base_gate(s[i], x[i], x[i + 1]) for i in range(n)]
    return recursive_circuit(remaining_s, layer_output)


def wrap_recursive_circuit(n):
    """
    Wrapper function to map recursive_circuit into a form that d-wave can accept
    :param n: layer depth for circuit
    :return:
    """
    n_s = int(n * (n + 1) / 2)
    n_x = n + 1

    s = ["s{}".format(i) for i in range(n_s)]
    x = ["x{}".format(i) for i in range(n_x)]

    def fun_of_args(*args):
        if len(args) != n_s + n_x:
            raise ValueError("incorrect number of arguments")
        s_vars = args[0:n_s]
        x_vars = args[n_s:-1]
        return recursive_circuit(s_vars, x_vars)

    return fun_of_args, s, x


def get_ns_nx(n):
    n_s = n * (n + 1) / 2
    n_x = n + 1
    return int(n_s), int(n_x)


def get_random_bits(n):
    return [bool(random.getrandbits(1)) for i in range(n)]


def split_and_add(l):
    """
    :param l: list of lists of booleans
    :return:
    """
    l_1 = [l_i + [0] for l_i in l]
    l_2 = [l_i + [1] for l_i in l]
    return l_1 + l_2


def make_specific_circuit(s):
    return lambda x: recursive_circuit(s, x)


def make_complete_data(specific_circuit, n):
    x_data = list(itertools.product([False, True], repeat=n + 1))

    y_data = []
    for row in x_data:
        y_data.append(specific_circuit(row))

    return x_data, y_data


def wrap_with_complete_data(specific_circuit, n):
    x_data, y_data = make_complete_data(specific_circuit, n)

    def output(*args):
        for (x_row, y) in zip(x_data, y_data):
            if recursive_circuit(args, x_row) != y:
                return False
        return True

    return output

def check_circuits_equivalent(weights_1, weights_2, n):
    x_data = list(itertools.product([False, True], repeat=n + 1))

    circuit_1 = make_specific_circuit(weights_1)
    circuit_2 = make_specific_circuit(weights_2)

    return all(circuit_1(x_row) == circuit_2(x_row) for x_row in x_data)


def make_base_polynomial(y, z_1, z_2, s):
    """
    Makes a polynomial of form Y + Z_1, + Z_2 + 2 Y S - 2 Y Z_1 - 2 Y Z_2 - S Z_1 -  S Z_2 + Z_1 Z_2 ,
     which represents the base_gate as a polynomial.
    inputs: y, z_1, z_2, s are all strings which represent variable names which will be put into a binary quadratic model.
    :return: dict of tuples to values
    """
    return {(y,): 1, (z_1,): 1, (z_2,): 1, (y, z_1): -2, (y, z_2): -2, (y, s): 2, (z_1, s): -1, (z_2, s): -1,
            (z_1, z_2): 1}

def make_output_polynomial(y_val, z_1, z_2, s):
    """
    Makes a polynomial as above which contains the restriction that y is equal to a certain value. ,
    inputs: z_1, z_2, s are all strings which represent variable names which will be put into a binary quadratic model.
    y_val is the actual value of y.
    :return: dict of tuples to values
    """

    if not y_val:
        return {(z_1,): 1, (z_2,): 1, (z_1, s): -1, (z_2, s): -1, (z_1, z_2): 1}
    else:
        return {(z_1,): -1, (z_2,): -1, (s,): 2, (z_1, s): -1, (z_2, s): -1, (z_1, z_2): 1}

def make_input_polynomial(y, z_1_val, z_2_val, s):
    """
    Makes a polynomial as above which contains the restriction that z_1 and z_2 are equal to a certain value.
    inputs: y, z_1, z_2, s are all strings which represent variable names which will be put into a binary quadratic model.
    :return: dict of tuples to values
    """

    if (not z_1_val) and (not z_2_val):
        return {(y,): 1, (y, s): 2}
    if z_1_val and z_2_val:
        return {(y,): -3,  (y, s): 2, (s,): -2}

    return {(y,): -1,  (y, s): 2, (s, ): -1}

def _merge_dicts_and_add(dict1, dict2):
    output_dict = { key: value for key, value in dict1.items() }
    for key, value in dict2.items():
        output_dict[key] = output_dict.get(key, 0) + value

    return output_dict

def merge_dicts_and_add(*args):
    output = {}
    for d in args:
        output = _merge_dicts_and_add(output, d)
    return output

def make_polynomial_for_datapoint(y_val, x_vals, z_start=0):
    if len(x_vals) == 2:
        return {  ('s_0',): 2 * y_val, ('s_0',): -1 * x_vals[0], ('s_0',): -1 * x_vals[1] }, 0
    if len(x_vals) < 2:
        raise ValueError("Please input a non-trivial amount of x values")

    polynomial = {}

    # First layer
    layer = ["z_{}".format(i + z_start) for i in range(len(x_vals)-1)]
    s_vals = ["s_{}".format(i) for i in range(len(x_vals)-1)]
    auxiliary_bit_tally = len(layer) + z_start
    s_bit_tally = len(s_vals)
    for x_i, x_j, z_i, s in zip(x_vals[0:-1], x_vals[1:], layer, s_vals):
        polynomial = merge_dicts_and_add(polynomial, make_input_polynomial(z_i, x_i, x_j, s))

    # Middle layers
    while len(layer) > 2:
        next_layer = ["z_{}".format(i + auxiliary_bit_tally) for i in range(len(layer)-1)]
        s_vals = ["s_{}".format(i + s_bit_tally) for i in range(len(layer)-1)]
        for y, x_i, x_j, s in zip(next_layer, layer[0:-1], layer[1:], s_vals):
            polynomial = merge_dicts_and_add(polynomial, make_base_polynomial(y, x_i, x_j, s))

        layer = next_layer
        auxiliary_bit_tally += len(next_layer)
        s_bit_tally += len(s_vals)


    # End layer
    z_1 = layer[0]
    z_2 = layer[1]
    s = "s_{}".format(s_bit_tally)
    polynomial = merge_dicts_and_add(polynomial, make_output_polynomial(y_val, z_1, z_2, s))
    return polynomial, auxiliary_bit_tally


def make_polynomial_for_many_datapoints(y_vals, x_vals):
    """ Same as above but x_vals and y_vals are lists """
    if len(y_vals) != len(x_vals):
        raise ValueError("x and y data don't match")
    if len(set(len(x) for x in x_vals)) != 1:
        raise ValueError("x data is not all same length")
    if len(x_vals[0]) < 2:
        raise ValueError("please enter at least two x values")

    polys = []
    z_start = 0
    for y, x_row in zip(y_vals, x_vals):
        poly, z_start = make_polynomial_for_datapoint(y, x_row, z_start)
        polys += [poly]

    return merge_dicts_and_add(*polys)

def get_interaction_variables(n_layers, batch_size):
    ## TODO: finish this and establish if its even worthwhile.
    n_s = ( n_layers * (n_layers + 1) ) / 2
    n_z_per_batch = n_s - 1
    s_vals = [ "s_{}".format(i) for i in range(n_s)]

    # First layer
    layer = ["z_{}".format(i) for i in range(n_layers)]
    s_vals = ["s_{}".format(i) for i in range(n_layers)]

    auxiliary_bit_tally = n_layers
    s_bit_tally = n_layers
    interactions = set()
    for z_i, s in zip(layer, s_vals):
        interactions.add((z_i, s))

    # Middle layers
    while len(layer) > 2:
        next_layer = ["z_{}".format(i + auxiliary_bit_tally) for i in range(len(layer)-1)]
        s_vals = ["s_{}".format(i + s_bit_tally) for i in range(len(layer)-1)]
        for y, x_i, x_j, s in zip(next_layer, layer[0:-1], layer[1:], s_vals):
            interactions.add(x_i, s)
            interactions.add(x_j, s)
            interactions.add(x_i, x_j)
            interactions.add(y, s)
            interactions.add(y, x_i)
            interactions.add(y, x_j)

        layer = next_layer
        auxiliary_bit_tally += len(next_layer)
        s_bit_tally += len(s_vals)


    # End layer
    z_1 = layer[0]
    z_2 = layer[1]
    s = "s_{}".format(s_bit_tally)
    polynomial = merge_dicts_and_add(polynomial, make_output_polynomial(y_val, z_1, z_2, s))
    return polynomial, auxiliary_bit_tally



def make_bqm(polynomial, offset):
    """ polynomial is a dictionary with tuples as keys"""

    linear = {}
    quadratic = {}

    for key, value in polynomial.items():
        if len(key) == 1:
            linear[key[0]] = value
        else:
            quadratic[key] = value

    return dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype=dimod.BINARY)
