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

def generate_random_weights(n_s):
    out = []
    for i in range(n_s):
        out += [random.choice([True, False])]
    return out

def sample_size_schedule(n_layers, n_batches, max_reads):
    """ The size of solution space grows proportionally to 2 ^ (n ^ 2 + n)/2.
        Increasing the number of samples proportionally to this should, as a rough measure,
        ensure that we get overlap amongst our partial solution sets.
    """
    scaling = int(2 ** (((n_layers ** 2) + n_layers) / 2)) * ( n_batches ** 1)
    return min(scaling, max_reads)

def num_batches_schedule(n_layers):
    if n_layers < 4:
        return [1, 2, 4]
    if n_layers == 4:
        return [2, 4, 8]
    elif n_layers == 5:
        return [8, 16]
    elif n_layers == 6:
        return [16, 32]
    else:
        return [] # embedding with a reasonable batch size is not feasible at 7 or more layers

# TODO: should I also do a search through different chain strengths?
# TODO: try extended J range?
# Int is not iterable??
# ============================================================== #
max_layers = 4
n_embedding_tries = 20
n_problems_per_size = 10
max_reads_per_batch = 1024
use_real_dwave = True

# ============================================================== #

if use_real_dwave:
    sampler = DWaveSampler()
    record_type = "dwave"
else:
    sampler = neal.SimulatedAnnealingSampler()
    record_type = "simulated"

output = np.zeros((max_layers - 1, max_layers + 1))
n_vars_array = np.zeros((max_layers - 1, max_layers + 1))

dirname = os.path.dirname(__file__) + "/battle_results"
records_file = dirname + "/records.pickle"
if os.path.exists(records_file):
    os.remove(records_file)

records = []
for n_layers in range(1, max_layers):
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
        brute_force_record = Record("brute_force", n_layers, weights, result, end-start)
        records += [brute_force_record]

        print("Beginning with dwave solvers .....")
        for n_batches in num_batches_schedule(n_layers):  # with batch, I want to take actual batch sizes of size 2 ** batch. there is N_layers +1 x vars which means range(n_layers + 2 hits that)
            n_reads_per_batch = sample_size_schedule(n_layers, n_batches, max_reads_per_batch)
            batch_size = int(2 ** (n_layers + 1) / n_batches)
            print("batch size: {}, n_batches: {}, reads per batch: {}".format(batch_size, n_batches, n_reads_per_batch))
            try:
                strategy = strat.SmarterStrategy(n_layers, n_embedding_tries, sampler, n_batches)
                start = time.time()
                solutions = strategy.solve(x_data, y_data, chain_strength=2.0, num_reads=n_reads_per_batch)
                end = time.time()
                embedding_time = strategy.embedding_time
                failure = len(solutions) == 0
                timing = strategy.timing
                record = Record(record_type, n_layers, weights, solutions, end-start, batch_size, n_batches, embedding_time, timing, failure)
            except Exception as e:
                print(e)
                record = Record(record_type, n_layers, weights, [], 0, batch_size, 0, {}, True, str(e))
            finally:
                records += [record]

            pickle.dump(records, open(records_file, "wb"))


