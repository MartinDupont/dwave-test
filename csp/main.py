import dwavebinarycsp
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import re
import random

# ====================================================================================== #
run_new = False
version = 16 ## only used when reading old versions
n_reads = 100
anneal_time = 20 ## default is 20
chain_strength = 2.0 # default is 1.0
# ====================================================================================== #
# Represent the map as the nodes and edges of a graph
provinces = ['AB', 'BC', 'MB', 'NB', 'NL', 'NS', 'NT', 'NU', 'ON', 'PE', 'QC', 'SK', 'YT']
neighbors = [('AB', 'BC'), ('AB', 'NT'), ('AB', 'SK'), ('BC', 'NT'), ('BC', 'YT'), ('MB', 'NU'),
             ('MB', 'ON'), ('MB', 'SK'), ('NB', 'NS'), ('NB', 'QC'), ('NL', 'QC'), ('NT', 'NU'),
             ('NT', 'SK'), ('NT', 'YT'), ('ON', 'QC')]

# Function for the constraint that two nodes with a shared edge not both select one color
def not_both_1(v, u):
    return not (v and u)

# Function that plots a returned sample
def plot_map(sample):
    G = nx.Graph()
    G.add_nodes_from(provinces)
    G.add_edges_from(neighbors)
    # Translate from binary to integer color representation
    color_map = {}
    for province in provinces:
        for i in range(colors):
            if sample[province+str(i)]:
                color_map[province] = i
    # Plot the sample with color-coded nodes
    node_colors = [color_map.get(node) for node in G.nodes()]
    nx.draw_circular(G, with_labels=True, node_color=node_colors, node_size=3000, cmap=plt.cm.rainbow)
    plt.show()

dirname = os.path.dirname(__file__)
results_path = dirname + "/results"

def find_new_lowest_version():
    files = os.listdir(results_path)
    hits = [f.replace("result", "").replace(".pickle", "") for f in files if re.match(r"result[0-9]*.pickle", f)]
    return max(int(h) for h in hits) + 1


# Valid configurations for the constraint that each node select a single color
one_color_configurations = {(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)}
colors = len(one_color_configurations)

# Create a binary constraint satisfaction problem
csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)

# Add constraint that each node (province) select a single color
for province in provinces:
    variables = [province+str(i) for i in range(colors)]
    csp.add_constraint(one_color_configurations, variables)

# Add constraint that each pair of nodes with a shared edge not both select one color
for neighbor in neighbors:
    v, u = neighbor
    for i in range(colors):
        variables = [v+str(i), u+str(i)]
        csp.add_constraint(not_both_1, variables)

# Convert the binary constraint satisfaction problem to a binary quadratic model
bqm = dwavebinarycsp.stitch(csp)

print(bqm)
find_new_lowest_version()


if run_new:
    # Sample n times
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample(bqm, num_reads=n_reads, annealing_time=anneal_time, chain_strength=chain_strength)
    new_version = find_new_lowest_version()
    pickle.dump(response, open('{}/result{}.pickle'.format(results_path, new_version), "wb"))
else:
    response = pickle.load(open('{}/result{}.pickle'.format(results_path, version), "rb" ))


sample = next(response.samples())

not_yet_plotted = True
failed = 0
for sample in response.samples():
    if not csp.check(sample):
        failed += 1
    else:
        print('=======================================================================================')
        print(sample)
        if not_yet_plotted:
            plot_map(sample)
            not_yet_plotted = False

print("Number of failed responses: {} out of {}".format(failed, len(response.samples())))

