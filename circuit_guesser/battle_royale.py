import strategies as strat
import samplers
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import circuits as c
from dwave.system.samplers import DWaveSampler
import neal
import time


class Record:
    def __init__(self, strategy, layers, actual_weights, solutions, running_time, embedding_time=0, sampling_time=0, failure=False):
        self.strategy = strategy
        self.layers = layers
        self.actual_weights = actual_weights
        self.solutions = solutions
        self.running_time = running_time
        self.embedding_time = embedding_time
        self.sampling_time = sampling_time
        self.failure = failure

def generate_random_weights(n_s):
    # TODO: implement
    return []

# ============================================================== #
max_layers = 6
n_embedding_tries = 1
n_problems_per_size = 4
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
        for batch in range(0, n_layers + 2): # with batch, I want to take actual batch sizes of size 2 ** batch. there is N_layers +1 x vars which means range(n_layers + 2 hits that)
            n_batches = 2 ** (n_layers + 1 - batch)
            strategy = strat.SmarterStrategy(n_layers, n_embedding_tries, 100, weights, sampler, n_batches)
            try:
                print("batch_size: {}".format(n_layers, 2 ** batch))
                start = time.time()
                solutions = strategy.solve()
                end = time.time()
                record = Record(record_type, n_layers, weights, solutions, end-start, "???", )
            except Exception as e:
                print(e)
                record = Record(record_type, n_layers, weights, [], 0, 0, 0, True)
            finally:
                records += [record]

            pickle.dump(records, open(records_file, "wb"))


plt.imshow(output)
plt.colorbar()
plt.title('worst chain length for embedding')
plt.ylabel('number of layers')
plt.yticks(range(max_layers - 1), range(1, max_layers + 1))
plt.xlabel('batch size')
plt.xticks(range(max_layers + 1), [ 2 ** i for i in range(max_layers + 1)])
plt.show()
plt.savefig(dirname + '/embeddings.png')
plt.close()


plt.imshow(n_vars_array)
plt.colorbar()
plt.title('number of total variables')
plt.ylabel('number of layers')
plt.yticks(range(max_layers - 1), range(1, max_layers + 1))
plt.xlabel('batch size')
plt.xticks(range(max_layers + 1), [ 2 ** i for i in range(max_layers + 1)])
plt.show()
plt.savefig(dirname + '/num_variables.png')
plt.close()

