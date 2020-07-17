import hybrid
from dimod import SampleSet

# Here is a list of the most frustrating problems I ran into while working with the Dwave.
# I was using it around january this year, so it could be that many of these problems have been fixed in the meantime.

# ====================================================== Issue 1 ===================================================== #
# Example issue. I wanted to run a really simple hybrid workflow.
# I want to break a problem into subproblems and solve them individually.
# Naively, I would make something like this:

subproblem = hybrid.EnergyImpactDecomposer(size=30, rolling_history=1.0)
subsampler = hybrid.SimulatedAnnealingSubproblemSampler()

iteration = subproblem | subsampler | hybrid.SplatComposer()
workflow = hybrid.LoopUntilNoImprovement(iteration, convergence=10)

# Unfortunately, this breaks! And it is really hard to figure out why.
# The error message is ValueError: mismatch between variables in 'initial_states' and 'bqm'
# It took me a while to figure out that the subsamples weren't being carried over correctly between different runs.
# Every cycle, the sampler would be given a new subproblem, but the subsamples would still be those from the old subproblem.
# So I instead had to program a fix for it:

def fix_dwave_bug(_, state, **kwargs):
    new_subsamples = []
    if not state.subsamples:
        return state
    for sample in state.subsamples:
        new_values = {}
        for variable in state.subproblem:
            try:
                value = sample[variable]
            except:
                value = state.samples.first.sample[variable]
            new_values[variable] = value
        new_subsamples += [new_values]

    new_subsamples = SampleSet.from_samples_bqm(new_subsamples, state.subproblem)
    return state.updated(subsamples = new_subsamples)


bridge = hybrid.Lambda(fix_dwave_bug)

iteration = subproblem | bridge | subsampler | hybrid.SplatComposer()

# Now it works!
# But to solve this I had to do a lot of digging around in the source code.

# ====================================================== Issue 2 ===================================================== #
# I lost a full day to this issue.
# In my code, I tried to first find an optimal embedding for my problem onto the D-Wave, and then solve the problem.
# So I had a loop that looked like this:


import dwave_networkx as dnx
import minorminer

def make_embedding(self):
    bqm = self.get_bqm() # some function which makes my BQM
    edges = [tup for tup in bqm.keys() if len(tup) == 2]
    if len(edges) == 0:
        return {'s_0': self.target_graph[0][0]}  # no edges, trivial problem, put the single qubit anywhere.

    best_chain_length = 10000000
    best_embedding = {}
    for i in range(self.n_embedding_tries):
        embedding = minorminer.find_embedding(edges, dnx.chimera_graph(16)) # Here's the mistake!
        if len(embedding.keys()) == 0:
            continue  # minorminer returns an empty dict when it cant find an embedding
        max_chain_length = max(len(value) for value in embedding.values())
        if max_chain_length < best_chain_length:
            best_chain_length = max_chain_length
            best_embedding = embedding

    if len(best_embedding.keys()) == 0:
        raise RuntimeError("No embedding found")
    return best_embedding

# But the problem is, when I tried to run my problem on the Dwave with a fixed embedding composite for the embedding I just calculated, it would break!

# The problem was, that the graph that's available on the Dwave isn't a perfect Chimera graph, and the embedding was calculated assuming that it was.
# This is actually understandable, considering that manufacturing errors can lead to broken bits etc.
# However, the problem was that this wasn't mentioned ANYWHERE in the documentation.
# And the error messages weren't at all helpful: ValueError('{} is not a connected chain'.format(chain))
# And the worst part was that the problem only showed up when I was running on the real QPU.
# I wasn't able to reproduce it locally, because the FixedEmbeddingComposite only accepts a DWaveSolver as argument.
# All these factors made the problem very hard to debug, and I had to spend a lot of time digging around in the source
# code until I figured out what the problem was.

# ====================================================== Issue 3 ===================================================== #

# The 'tries' parameter in minorminer.find_embedding doesn't appear to do anything
# In issue 2, I wrote a loop which tried to find the optimal embedding in terms of maximum chain length
# I had thought that the 'tries' parameter would do this for me. The default is 10, but when I set the value even
# to 10,000 the algorithm didn't take any longer to run.

edges = {} # Some set of edges
minorminer.find_embedding(edges, dnx.chimera_graph(16), tries=10000)

# Furthermore, I looked in the source code, and the 'tries' variable doesn't actually appear to be used anywhere:
# https://github.com/dwavesystems/minorminer/blob/main/minorminer/_minorminer.pyx
# Although I could simply not be seeing it.

# Although, it isn't clear what the 'tries' parameter actually should do, based on the documentation.
# Does it keep retrying until it finds a solution, or does it keep retrying until it finds the best solution?
# If it doesn't try and find the best solution, then I would recommend that the loop I wrote in Issue 2 somehow makes
# it's way into the standard library. It strikes me as a common use-case that people would want to find embeddings
# that minimize the chain length.

# ====================================================== Issue 4 ===================================================== #
# QBSolv
# Firstly, the documentation is incorrect, that confused me for an hour or so
# https://github.com/dwavesystems/qbsolv/blob/master/python/dwave_qbsolv/dimod_wrapper.py
# The documentation states that the argument 'Q' of the 'sample' method needs to be a dictionary.
# This is clearly not true, as in line 93 you call '.to_qubo()' on it, which means that it actually needs to be a BinaryQuadraticModel

# Now, I read somewhere that this project is deprecated, and we should prefer using dwave-hybrid. But the problem is,
# dwave-hybrid doesn't have the QBsolv algorithm, it only has a 'simplified' version which is actually radically different
# to the QBSolv algorithm. I'm not sure where one should go in order to use the QBSolv algorithm.

# ======================================================= Notes ====================================================== #

# The ocean software tries to have all of its different solvers and embedding composites fit nicely together like lego,
# with a nice object-oriented class heirachy that makes everything simple and replaceable etc.
# Unfortunately, the execution isn't that great. Issue 1 and 2 really highlight what I'm talking about.
# I had some classes which I wanted to stick together, and I had the impression that I could, but instead they threw
# really weird errors that didn't immediately tell me I have actually composed two things which I shouldn't have.

# I hope you find these notes helpful
