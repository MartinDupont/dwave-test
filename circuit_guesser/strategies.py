import itertools
import math
import time

import circuits as c
import dwave_networkx as dnx
import dwavebinarycsp
import minorminer
import neal
from dwave.system import FixedEmbeddingComposite, DWaveSampler
from dwave_qbsolv import QBSolv


def get_max_batch_size(n_batches, n_xs):
    num_rows = 2 ** n_xs
    return math.ceil(num_rows / n_batches)


class BaseStrategy:
    """ Really just contains helper methods for the other classes which inherit from it"""

    def check_solution_for_row(self, x_data, y_data, solution):
        # This is brittle. we depend on the convention that s is solutions.

        vals = [solution[k] for k in sorted(solution.keys()) if k[0] == "s"]
        circuit = c.make_specific_circuit(vals)
        return circuit(x_data) == (not not y_data)

    def check_multiple_solutions(self, x_rows, y_rows, sample):
        for x_row, y in zip(x_rows, y_rows):
            if not self.check_solution_for_row(x_row, y, sample):
                return False
        return True

    def convert_dict_to_tuples(self, input_dict):
        return tuple([(a, b) for a, b in input_dict.items()])

    def strip_auxiliary_variables(self, input_dict):
        return {a: b for a, b in input_dict.items() if a[0] == "s"}

    def convert_tuples_to_dict(self, tuples):
        return {key: value for key, value in tuples}

    def solve(self, x_data, y_data, **kwargs):
        return []


class CspStrategy:
    def __init__(self, n_layers, circuit_weights, sampler):
        self.circuit_weights = circuit_weights
        self.sampler = sampler
        self.n_layers = n_layers

    def solve(self, x_data, y_data, **kwargs):
        constraint_satisfaction_problem = c.wrap_with_data(x_data, y_data)

        weight_variable_names = ["s{}".format(i) for i in range(len(self.circuit_weights))]

        csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)
        csp.add_constraint(constraint_satisfaction_problem, weight_variable_names)

        bqm = dwavebinarycsp.stitch(csp)
        return self.sampler.sample(bqm, **kwargs)


class BatchStrategy(BaseStrategy):
    def __init__(self, n_layers, n_embedding_tries, sampler, n_batches):
        self.n_layers = n_layers
        self.n_embedding_tries = n_embedding_tries
        self.n_batches = n_batches
        self.timing = {}

        if isinstance(sampler, DWaveSampler):
            self.target_graph = sampler.edgelist
            start = time.time()
            embedding = self.make_embedding()
            end = time.time()
            self.sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
            self.embedding_time = end - start
        else:
            self.sampler = sampler
            self.embedding_time = 0
            self.target_graph = dnx.chimera_graph(16)

    def get_most_complex_polynomial(self):
        x_row = [1 for i in range(self.n_layers + 1)]
        n_rows = get_max_batch_size(self.n_batches, self.n_layers + 1)
        x_rows = [x_row for i in range(n_rows)]  ## this is a hack. x_rows with all 1s will never occur.
        y_rows = [1 for i in range(n_rows)]
        bqm = c.make_polynomial_for_many_datapoints(y_rows, x_rows)  # the polynomial with all 1s contains all the edges
        return bqm

    def make_embedding(self):
        bqm = self.get_most_complex_polynomial()
        edges = [tup for tup in bqm.keys() if len(tup) == 2]
        if len(edges) == 0:
            return {'s_0': self.target_graph[0][0]}  # no edges, trivial problem, put the single qubit anywhere.

        best_chain_length = 10000000
        best_embedding = {}
        for i in range(self.n_embedding_tries):
            embedding = minorminer.find_embedding(edges, self.target_graph)
            if len(embedding.keys()) == 0:
                continue  # minorminer returns an empty dict when it cant find an embedding
            max_chain_length = max(len(value) for value in embedding.values())
            if max_chain_length < best_chain_length:
                best_chain_length = max_chain_length
                best_embedding = embedding

        if len(best_embedding.keys()) == 0:
            keys = set(k for tup in bqm.keys() for k in tup)
            raise RuntimeError("No embedding found for problem with {} variables".format(len(keys)))
        return best_embedding

    def solve_batch(self, x_rows, y_rows, **kwargs):
        offset = 0
        poly = c.make_polynomial_for_many_datapoints(y_rows, x_rows)
        bqm = c.make_bqm(poly, offset)
        result = self.sampler.sample(bqm, **kwargs)
        self.timing = c._merge_dicts_and_add(self.timing, result.info.get('timing', {}))
        solution_set = set()
        for datum in result.data(['sample', 'energy', 'num_occurrences']):
            solution = self.convert_dict_to_tuples(self.strip_auxiliary_variables(datum.sample))

            if self.check_multiple_solutions(x_rows, y_rows, datum.sample):
                solution_set.add(solution)
        return solution_set

    def solve(self, x_data, y_data, **kwargs):
        """This guy will be smarter and process the polynomials in batches,
         then take the intersection over all in order to find a solution.
         We let the thingy do the embedding for us"""

        # for each batch, solve the problem.
        first = True
        for i in range(self.n_batches):
            x_rows = x_data[i::self.n_batches]
            y_rows = y_data[i::self.n_batches]

            solution_set = self.solve_batch(x_rows, y_rows, **kwargs)

            if first:
                final_solutions = solution_set
                first = False
            else:
                final_solutions = final_solutions.intersection(solution_set)

        return [self.convert_tuples_to_dict(sol) for sol in final_solutions]


class BruteForceStrategy:
    def __init__(self, n_layers):
        self.n_layers = n_layers

    def solve(self, x_data, y_data, **kwargs):

        constraint_satisfaction_problem = c.wrap_with_data(x_data, y_data)
        n_s, _ = c.get_ns_nx(self.n_layers)

        count = 0
        for s_vals in itertools.product([False, True], repeat=n_s):
            if count % 64 == 0:
                print("{} out of {} combinations tried".format(count, 2 ** n_s), end="\r"),
            if constraint_satisfaction_problem(s_vals):
                return s_vals
            count += 1

        raise RuntimeError("Brute force strategy ran to completion without finding a solution.")


class QBSolvStrategy(BaseStrategy):
    def __init__(self, sampler, n_tries=1):
        self.sampler = sampler
        self.n_tries = n_tries
        self.timing = {}

    def solve(self, x_data, y_data, **kwargs):
        offset = 0
        poly = c.make_polynomial_for_many_datapoints(y_data, x_data)
        bqm = c.make_bqm(poly, offset)
        valid_solutions = []
        give_up = False
        count = 0

        while not valid_solutions and not give_up:
            response = QBSolv().sample(bqm, solver=self.sampler)
            self.timing = c._merge_dicts_and_add(self.timing, response.info.get('timing', {}))
            for sample in response.samples():
                valid = self.check_multiple_solutions(x_data, y_data, sample)
                if valid:
                    valid_solutions += [self.strip_auxiliary_variables(sample)]
            count += 1
            give_up = count > self.n_tries
            print("QBSolv strategy tried {} times".format(count), end="\r")

        return valid_solutions


if __name__ == "__main__":
    n_layers = 3
    weights = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
    weights = [0, 0, 0, 0, 0, 0]
    sampler = neal.SimulatedAnnealingSampler()
    strategy = BatchStrategy(n_layers, 100, sampler, 4)
    # embedding = strategy.make_embedding()
    # print("-----------------------------------------------------------------")
    # print(embedding)
    circuit = c.make_specific_circuit(weights)
    x_data, y_data = c.make_complete_data(circuit, n_layers)
    results = strategy.solve(x_data, y_data, num_reads=1000)
    for r in results:
        print("===========================================")
        sorted_keys = sorted(r.keys())
        new_weights = [r[k] for k in sorted_keys]

        print("Correct: {}".format(c.check_circuits_equivalent(weights, new_weights, n_layers)))
        print(r)
