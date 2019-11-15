import dwavebinarycsp
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite


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


def recursive_circuit(s, x):
    """
    Recursive circuit
    :param s: list of control variables s (spins in dwave)
    :param x: list of inputs
    :return: single boolean
    """
    n_s = len(s)
    n_x = len(x)
    if (n_x * (n_x - 1)/2) != n_s:
        raise ValueError("number of s arguments must be compatible with number of x arguments")

    # need a base case
    if n_s <= 0:
        return x[0]

    remaining_s = s[n_x:-1]
    layer_output = [base_gate(s[i], x[i], x[i + 1]) for i in range(n_x - 1)]
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
