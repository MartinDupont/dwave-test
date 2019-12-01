import itertools

import circuits as c
import dwavebinarycsp
import dwave
import minorminer
import dwave_networkx as dnx
from dwave.system import FixedEmbeddingComposite, DWaveSampler
import neal
import math

def get_max_batch_size(n_batches, n_xs):
    num_rows = 2 ** n_xs
    return math.ceil(num_rows / n_batches)


class CspStrategy:
    def __init__(self, n_layers, circuit_weights, sampler):
        self.circuit_weights = circuit_weights
        self.sampler = sampler
        self.n_layers = n_layers


    def solve(self, **kwargs):


        actual_circuit = c.make_specific_circuit(self.circuit_weights)

        constraint_satisfaction_problem = c.wrap_with_complete_data(actual_circuit, self.n_layers)

        weight_variable_names = ["s{}".format(i) for i in range(len(self.circuit_weights))]

        csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)
        csp.add_constraint(constraint_satisfaction_problem, weight_variable_names)

        bqm = dwavebinarycsp.stitch(csp)
        return self.sampler.sample(bqm, **kwargs)


class EliminationStrategy:
    def __init__(self, n_layers, circuit_weights, sampler):
        self.circuit_weights = circuit_weights
        self.n_layers = n_layers

        if isinstance(sampler, neal.SimulatedAnnealingSampler):
            self.sampler = sampler
        else:
            embedding = self.make_embedding()
            self.sampler = FixedEmbeddingComposite(DWaveSampler(), embedding)
            
    def get_most_complex_polynomial(self):
        x_data = [1 for i in range(self.n_layers + 1)]
        bqm, _ = c.make_polynomial_for_datapoint(1, x_data) # the polynomial with all 1s contains all the edges
        return bqm
        
    def make_embedding(self):
        bqm = self.get_most_complex_polynomial()
        print(bqm)
        edges = [tup for tup in bqm.keys() if len(tup) == 2]
        chimera = dnx.chimera_graph(16)

        best_chain_length = 10000000
        for i in range(100):
            embedding = minorminer.find_embedding(edges, chimera)
            max_chain_length = max(len(value) for value in embedding.values())
            if max_chain_length < best_chain_length:
                best_chain_length = max_chain_length
                print(best_chain_length)
                best_embedding = embedding
        return best_embedding

    def check_solution(self, x_data, y_data, solution):
        # This is brittle. we depend on the convention that s is solutions.

        vals = [solution[k] for k in sorted(solution.keys()) if k[0] == "s"]
        circuit = c.make_specific_circuit(vals)
        return circuit(x_data) == (not not y_data)

    def convert_dict_to_tuples(self, input_dict):
        return tuple([(a, b) for a, b in input_dict.items() if a[0] == "s"])

    def convert_tuples_to_dict(self, tuples):
        return { key: value for key, value in tuples }

    def solve(self, **kwargs):
        """This guy will be sortof dumb and find a set of possible solutions for EACH datapoint,
         then take the intersection over all in order to find a solution. We let the thingy do the embedding for us"""

        actual_circuit = c.make_specific_circuit(self.circuit_weights)

        # generate data.
        x_data, y_data = c.make_complete_data(actual_circuit, self.n_layers)

        offset = 0
        # for each row, solve the problem.
        individual_solutions = []
        for x_vec, y in zip(x_data, y_data):
            poly, _ = c.make_polynomial_for_datapoint(y, x_vec)
            bqm = c.make_bqm(poly, offset)
            result = sampler.sample(bqm, num_reads=100)
            solution_set = set()
            for datum in result.data(['sample', 'energy', 'num_occurrences']):
                solution = self.convert_dict_to_tuples(datum.sample)

                if self.check_solution(x_vec, y, datum.sample):
                    solution_set.add(solution)

            individual_solutions += [solution_set]

        final_solutions = individual_solutions[0]
        for sol in individual_solutions:
            final_solutions = final_solutions.intersection(sol)

        return [ self.convert_tuples_to_dict(sol) for sol in final_solutions]
        # Question: do we reject all high-energy solutions?
        
class SmarterStrategy(EliminationStrategy):
    def __init__(self, n_layers, circuit_weights, sampler, n_batches):
        super().__init__(n_layers, circuit_weights, sampler)
        self.n_batches = n_batches 
        
    def get_most_complex_polynomial(self):
        x_row = [1 for i in range(self.n_layers + 1)]
        n_rows = get_max_batch_size(self.n_batches, self.n_layers + 1)
        x_rows = [x_row for i in range(n_rows)]  ## this is a hack. x_rows with all 1s will never occur. 
        y_rows = [1 for i in range(n_rows)]
        bqm = c.make_polynomial_for_many_datapoints(y_rows, x_rows) # the polynomial with all 1s contains all the edges  
        return bqm
        
    def check_multiple_solutions(self, x_rows, y_rows, sample):
        for x_row, y in zip(x_rows, y_rows):
            if not self.check_solution(x_row, y, sample):
                return False
        return True
        
    def solve(self):
        """This guy will be smarter and process the polynomials in batches,
         then take the intersection over all in order to find a solution.
         We let the thingy do the embedding for us"""

        actual_circuit = c.make_specific_circuit(self.circuit_weights)

        # generate data.
        x_data, y_data = c.make_complete_data(actual_circuit, self.n_layers)

        offset = 0
        # for each row, solve the problem.
        individual_solutions = []
        for i in range(self.n_batches):
            x_rows = x_data[i::self.n_batches]
            y_rows = y_data[i::self.n_batches]
        
            poly = c.make_polynomial_for_many_datapoints(y_rows, x_rows)
            bqm = c.make_bqm(poly, offset)
            result = sampler.sample(bqm, num_reads=100)
            solution_set = set()
            for datum in result.data(['sample', 'energy', 'num_occurrences']):
                solution = self.convert_dict_to_tuples(datum.sample)

                if self.check_multiple_solutions(x_rows, y_rows, datum.sample):
                    solution_set.add(solution)

            individual_solutions += [solution_set]

        final_solutions = individual_solutions[0]
        for sol in individual_solutions:
            final_solutions = final_solutions.intersection(sol)

        return [ self.convert_tuples_to_dict(sol) for sol in final_solutions]



# TODO: check that I"m scaling my polynomials correctly!!!!!!!!!!!!!!!!!!!!
# TODO: make get_embedding more efficient
if __name__ == "__main__":
    n_layers = 4
    weights = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
    #weights = [1, 1, 1, 0, 0, 1]
    sampler = neal.SimulatedAnnealingSampler()
    strategy = SmarterStrategy(n_layers, weights, sampler, 4)
    embedding = strategy.make_embedding()
    print("-----------------------------------------------------------------")
    print(embedding)
    results = strategy.solve()
    for r in results:
        print("===========================================")
        sorted_keys = sorted(r.keys())
        new_weights = [r[k] for k in sorted_keys]
        
        print("Correct: {}".format(c.check_circuits_equivalent(weights, new_weights, n_layers)))
        print(r)
