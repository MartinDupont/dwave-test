import strategies as strat
import samplers
import numpy as np
import pickle
import os
import circuits as c
from dwave.system.samplers import DWaveSampler
import neal
import time
import random
from record import Record

SIMULATED_ANNEALING = 'SIMULATED_ANNEALING'
D_WAVE = 'D_WAVE'
BRUTE_FORCE = 'BRUTE_FORCE'


def generate_random_weights(n_s):
    out = []
    for i in range(n_s):
        out += [random.choice([True, False])]
    return out


def solve_with_strategy(sampler, n_tries, x_data, y_data, weights, record_type, n_layers):
    try:
        strategy = strat.QBSolvStrategy(sampler, n_tries)
        start = time.time()
        solutions = strategy.solve(x_data, y_data)
        end = time.time()
        failure = len(solutions) == 0
        timing = strategy.timing
        record = Record(record_type, n_layers, weights, solutions, end - start, timing=timing, failure=failure)
    except Exception as e:
        print(e)
        record = Record(record_type, n_layers, weights, [], 0, failure=True, failure_message=str(e))

    return record


# ============================================================== #
max_layers = 5
n_problems_per_size = 10
n_tries = 20
use_real_dwave = False

# ============================================================== #

simulated_sampler = neal.SimulatedAnnealingSampler()
dwave_sampler = DWaveSampler()

output = np.zeros((max_layers - 1, max_layers + 1))
n_vars_array = np.zeros((max_layers - 1, max_layers + 1))

dirname = os.path.dirname(__file__) + "/battle_results_qbsolv"
records_file = dirname + "/records.pickle"
if os.path.exists(records_file):
    os.remove(records_file)

records = []
for n_layers in range(2, max_layers):
    n_s, _ = c.get_ns_nx(n_layers)
    for problem in range(n_problems_per_size):
        weights = generate_random_weights(n_s)
        circuit = c.make_specific_circuit(weights)
        x_data, y_data = c.make_complete_data(circuit, n_layers)
        print("============================== New Problem ==============================")
        print("n_layers : {}, weights: {}".format(n_layers, weights))
        brute_force = strat.BruteForceStrategy(n_layers)
        start = time.time()
        result = brute_force.solve(x_data, y_data)
        end = time.time()
        brute_force_record = Record(BRUTE_FORCE, n_layers, weights, result, end - start)
        records += [brute_force_record]

        print('starting with simulated annealing solver....')
        record = solve_with_strategy(simulated_sampler, n_tries, x_data, y_data, weights, SIMULATED_ANNEALING, n_layers)
        records += [record]
        pickle.dump(records, open(records_file, "wb"))

        if use_real_dwave:
            print('starting with dwave sampler....')
            record = solve_with_strategy(dwave_sampler, n_tries, x_data, y_data, weights, D_WAVE, n_layers)
            records += [record]
            pickle.dump(records, open(records_file, "wb"))