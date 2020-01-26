import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
dirname = os.path.dirname(__file__) + "/embedding_results"

output = pickle.load(open('{}/embeddings.pickle'.format(dirname), "rb" ))
n_vars_array = pickle.load(open('{}/num_variables.pickle'.format(dirname), "rb" ))
times_array = pickle.load(open('{}/times.pickle'.format(dirname), "rb" ))

n_layers = len(output[:, 0])


plt.imshow(output)
plt.colorbar()
plt.title('worst chain length for embedding')
plt.ylabel('number of layers')
plt.yticks(range(n_layers - 1), range(1, n_layers + 1))
plt.xlabel('batch size')
plt.xticks(range(n_layers + 1), [ 2 ** i for i in range(n_layers + 1)])
plt.savefig(dirname + '/embeddings.png')
plt.show()
plt.close()


plt.imshow(n_vars_array)
plt.colorbar()
plt.title('number of total variables')
plt.ylabel('number of layers')
plt.yticks(range(n_layers - 1), range(1, n_layers + 1))
plt.xlabel('batch size')
plt.xticks(range(n_layers + 1), [ 2 ** i for i in range(n_layers + 1)])
plt.savefig(dirname + '/num_variables.png')
plt.show()
plt.close()

plt.imshow(times_array)
plt.colorbar()
plt.title('time to find embedding')
plt.ylabel('time [s]')
plt.yticks(range(n_layers - 1), range(1, n_layers + 1))
plt.xlabel('batch size')
plt.xticks(range(n_layers + 1), [ 2 ** i for i in range(n_layers + 1)])
plt.savefig(dirname + '/times.png')
plt.show()
plt.close()
