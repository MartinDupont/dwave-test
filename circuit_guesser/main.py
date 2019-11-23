import random
import dwavebinarycsp
import itertools
import circuits as c
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import neal
import os
import re
import pickle

def find_new_lowest_version(path):
    files = os.listdir(path)
    hits = [f.replace("result", "").replace(".pickle", "") for f in files if re.match(r"result[0-9]*.pickle", f)]
    if len(hits) == 0:
        return 0
    return max(int(h) for h in hits) + 1


# ==================================================================================================================== #

n_layers = 3
n_reads = 100
anneal_time = 20 ## default is 20
chain_strength = 1.0 # default is 1.0
run_new = False

# ==================================================================================================================== #

if __name__ == "__main__":
    n_circuit_weights, _ = c.get_ns_nx(n_layers)
    initial_circuit_weights = c.get_random_bits(n_circuit_weights)

    actual_circuit = c.make_specific_circuit(initial_circuit_weights)

    constraint_satisfaction_problem = c.wrap_with_complete_data(actual_circuit, n_layers)

    weight_variable_names = ["s{}".format(i) for i in range(n_circuit_weights)]

    csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)
    #csp.add_constraint(constraint_satisfaction_problem, weight_variable_names)
    csp.add_constraint(c.thingy, ['y', 's', 'x1', 'x2'])

    bqm = dwavebinarycsp.stitch(csp)
    print(bqm)
    if run_new:

        dirname = os.path.dirname(__file__)
        results_path = dirname + "/results"
        # Sample n times
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample(bqm, num_reads=n_reads, annealing_time=anneal_time, chain_strength=chain_strength)
        new_version = find_new_lowest_version(results_path)
        pickle.dump(response, open('{}/result{}.pickle'.format(results_path, new_version), "wb"))
        settings = {
            "n_reads": n_reads,
            "anneal_time": anneal_time,
            "chain_strength": chain_strength,
            "n_layers": n_layers,
            "circuit": initial_circuit_weights
        }
        pickle.dump(response, open('{}/settings{}.pickle'.format(results_path, new_version), "wb"))
    else:
        sampler = neal.SimulatedAnnealingSampler()

        response = sampler.sample(bqm, num_reads=n_reads)

    print("============================= New run ===============================")
    print("depth: {}".format(n_layers))
    print("weights: {}".format(initial_circuit_weights))
    print("=====================================================================")

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

