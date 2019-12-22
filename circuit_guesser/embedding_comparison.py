import strategies as strat
import samplers
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import circuits as c

# ============================================================== #
max_layers = 7
n_embedding_tries = 20

# ============================================================== #

sampler = samplers.MockSampler([])

output = np.zeros((max_layers - 1, max_layers + 1))
n_vars_array = np.zeros((max_layers - 1, max_layers + 1))

dirname = os.path.dirname(__file__) + "/embedding_results"
embeddings_file = dirname + "/embeddings.pickle"
if os.path.exists(embeddings_file):
    os.remove(embeddings_file)

variables_file = dirname + "/num_variables.pickle"
if os.path.exists(variables_file):
    os.remove(variables_file)

for n_layers in range(1, max_layers):
    n_s, _ = c.get_ns_nx(n_layers)
    weights = [1 for i in range(n_s)]
    for batch in range(0, n_layers + 2): # with batch, I want to take actual batch sizes of size 2 ** batch. there is N_layers +1 x vars which means range(n_layers + 2 hits that)
        n_batches = 2 ** (n_layers + 1 - batch )
        strategy = strat.SmarterStrategy(n_layers, n_embedding_tries, 100, weights, sampler, n_batches)
        try:
            print("===========================================================")
            print("n_layers : {}, batch_size: {}".format(n_layers, 2 ** batch))
            embedding = strategy.make_embedding()
            worst_chain_length = max(len(value) for value in embedding.values())
        except Exception as e: # errors are not exceptions!?!?!?!??!?
            print(e)
            worst_chain_length = np.inf
        finally:
            bqm = strategy.get_most_complex_polynomial()
            n_vars = len(set(k for tup in bqm.keys() for k in tup))

        output[n_layers - 1, batch] = worst_chain_length
        pickle.dump(output, open(embeddings_file, "wb"))

        n_vars_array[n_layers - 1, batch] = n_vars
        pickle.dump(n_vars_array, open(variables_file, "wb"))

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

