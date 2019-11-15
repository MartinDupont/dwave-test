import random
import dwavebinarycsp
from .circuits import recursive_circuit

def get_ns_nx(n):
    n_s = n * (n + 1) / 2
    n_x = n + 1
    return n_s, n_x

def get_random_bits(n):
    return [bool(random.getrandbits(1)) for i in n]

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
    x_data = []
    for i in range(n + 1):
        x_data = split_and_add(x_data)

    y_data = []
    for row in x_data:
        y_data.append(specific_circuit(row))

    return x_data, y_data


def wrap_with_complete_data(specific_circuit, n):
    x_data, y_data = make_complete_data(specific_circuit, n)

    def output(s):
        prod = True
        for (x_row, y) in zip(x_data, y_data):
            prod = prod & recursive_circuit(s, x_row) == y

    return output


# ==================================================================================================================== #

n_layers = 1

# ==================================================================================================================== #

n_circuit_weights, _ = get_ns_nx(n_layers)
initial_circuit_weights = get_random_bits(n_circuit_weights)

actual_circuit = make_specific_circuit(initial_circuit_weights)

constraint_satisfaction_problem = wrap_with_complete_data(actual_circuit, n_layers)

weight_variable_names = ["s{}".format(i) for i in range(n_circuit_weights)]

csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)
csp.add_constraint(constraint_satisfaction_problem, weight_variable_names)

bqm = dwavebinarycsp.stitch(csp)

