import circuits as c
import dwavebinarycsp


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
