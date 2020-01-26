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
    scaling = int(2 ** (((n_layers ** 2) + n_layers) / 2)) * ( n_batches ** 2)
    max_reads = 1024
    return min(scaling, max_reads)

# TODO: should I also do a search through different chain strengths?
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
        for batch in range(1, n_layers + 2): # with batch, I want to take actual batch sizes of size 2 ** batch. there is N_layers +1 x vars which means range(n_layers + 2 hits that)
            n_batches = 2 ** (n_layers + 1 - batch)
            n_reads_per_batch = sample_size_schedule(n_layers, n_batches, max_reads_per_batch)
            batch_size = 2 ** batch
            strategy = strat.SmarterStrategy(n_layers, n_embedding_tries, sampler, n_batches)
            print("batch size: {}".format(batch_size))
            try:
                start = time.time()
                solutions = strategy.solve(x_data, y_data, chain_strength=2.0, num_reads=n_reads_per_batch)
                end = time.time()
                embedding_time = strategy.embedding_time
                failure = len(solutions) == 0
                timing = strategy.timing
                record = Record(record_type, n_layers, weights, solutions, end-start, batch_size, embedding_time, timing, failure)
            except Exception as e:
                print(e)
                record = Record(record_type, n_layers, weights, [], 0, batch_size, 0, {}, True)
            finally:
                records += [record]

            pickle.dump(records, open(records_file, "wb"))


