from src.graphs.StructuralEquations import StringEquation
import numpy as np


class Node:
    def __init__(self, name, parents_name, children_name, equation, variable_cost, fixed_cost, min_intervention=None,
                 max_intervention=None, seed=0):
        self._name = name
        self._parents_name = parents_name
        self._children_name = children_name
        self._equation = equation
        self._fixed_cost = fixed_cost
        self._variable_cost = variable_cost
        self._min_intervention = min_intervention
        self._max_intervention = max_intervention
        self.parents = []
        self.children = []
        self.value = None
        self._seed = seed

    @property
    def name(self):
        return self._name

    @property
    def parents_name(self):
        return self._parents_name

    @property
    def children_name(self):
        return self._children_name

    def total_cost(self, interventions):
        return self._fixed_cost + self._variable_cost(interventions)

    def structural_equation(self):
        if isinstance(self._equation, StringEquation):
            # We allow named parameters here so that the lambdas can use the node names as variable names
            parent_values = {p.name: p.value for p in self.parents}
            self.value = np.float64(self._equation.predict(**parent_values))
        else:
            parent_values = np.array([[p.value for p in self.parents]])
            self.value = np.float64(self._equation.predict(parent_values))

    def fit_equation(self, node_measurement=None, parents_measurements=None):
        if parents_measurements is None:
            self._equation.fit(node_measurement)
        else:
            # if both measurements are None, fit has no effect
            self._equation.fit(parents_measurements, node_measurement)

