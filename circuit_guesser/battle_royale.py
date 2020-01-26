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

# TODO: should I also do a search through different chain strengths?
# TODO: maybe play around with the number of tries I take when running my samplers? (should yield better overlap)
# ============================================================== #
max_layers = 5
n_embedding_tries = 1
n_problems_per_size = 10
use_real_dwave = False

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
        print("============================== New Problem ==============================")
        print("n_layers : {}, weights: {}".format(n_layers, weights))
        brute_force = strat.BruteForceStrategy(n_layers, weights)
        start = time.time()
        result = brute_force.solve()
        end = time.time()
        brute_force_record = Record("brute_force", n_layers, weights, result, end-start)
        records += [brute_force_record]

        print("Beginning with dwave solvers .....")
        for batch in range(1, n_layers + 2): # with batch, I want to take actual batch sizes of size 2 ** batch. there is N_layers +1 x vars which means range(n_layers + 2 hits that)
            n_batches = 2 ** (n_layers + 1 - batch)
            batch_size = 2 ** batch
            strategy = strat.SmarterStrategy(n_layers, n_embedding_tries, 100, weights, sampler, n_batches)
            print("batch size: {}".format(batch_size))
            try:
                start = time.time()
                solutions = strategy.solve(chain_strength=2.0)
                end = time.time()
                embedding_time = strategy.embedding_time
                failure = len(solutions) == 0
                timing = strategy.timing
                record = Record(record_type, n_layers, weights, solutions, end-start, batch_size, embedding_time, timing, failure)
            except Exception as e:
                print(e)
                record = Record(record_type, n_layers, weights, [], 0, 0, {}, True)
            finally:
                records += [record]

            pickle.dump(records, open(records_file, "wb"))


