import numpy as np
from emukit.core.acquisition import Acquisition


class Cost(Acquisition):

    def __init__(self, costs_functions, evaluated_set):
        self.costs_functions = costs_functions
        self.evaluated_set = evaluated_set

    def evaluate(self, x):
        cost = 0
        for i in range(len(self.evaluated_set)):
            # TODO Test if the following works
            #  as one of them was just self.costs_functions[self.evaluated_set[i]](x) without [:, i]
            cost += self.costs_functions[self.evaluated_set[i]](x[:, i])
        return cost

    @property
    def has_gradients(self):
        return True
    
    def evaluate_with_gradients(self, x):
        return self.evaluate(x), np.zeros(x.shape)


def total_cost(intervention_variables, costs, x_new_dict):
    total_cost = 0.
    for i in range(len(intervention_variables)):
        total_cost += costs[intervention_variables[i]](x_new_dict[intervention_variables[i]])
    return total_cost
