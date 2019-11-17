import random
import dwavebinarycsp
import itertools
import circuits as c
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import neal




# ==================================================================================================================== #

n_layers = 3

# ==================================================================================================================== #

if __name__ == "__main__":
    n_circuit_weights, _ = c.get_ns_nx(n_layers)
    initial_circuit_weights = c.get_random_bits(n_circuit_weights)

    actual_circuit = c.make_specific_circuit(initial_circuit_weights)

    constraint_satisfaction_problem = c.wrap_with_complete_data(actual_circuit, n_layers)

    weight_variable_names = ["s{}".format(i) for i in range(n_circuit_weights)]

    csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)
    csp.add_constraint(constraint_satisfaction_problem, weight_variable_names)

    bqm = dwavebinarycsp.stitch(csp)

    sampler = neal.SimulatedAnnealingSampler()

    response = sampler.sample(bqm, num_reads=1000)

    print("============================= New run ===============================")
    print("depth: {}".format(n_layers))
    print("weights: {}".format(initial_circuit_weights))
    print("=====================================================================")

    solution = {'s0': 0, 's1': 1, 's2': 1}
    csp.check(solution)
    already_printed = False
    # Check how many solutions meet the constraints (are valid)
    valid, invalid, data = 0, 0, []
    for datum in response.data(['sample', 'energy', 'num_occurrences']):
        if (csp.check(datum.sample)):
            if not already_printed:
                print(datum.sample)
                already_printed = True
            valid = valid+datum.num_occurrences
            for i in range(datum.num_occurrences):
                data.append((datum.sample, datum.energy, '1'))
        else:
            invalid = invalid+datum.num_occurrences
            for i in range(datum.num_occurrences):
                data.append((datum.sample, datum.energy, '0'))
    print(valid, invalid)

