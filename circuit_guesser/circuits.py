import itertools
import random


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
