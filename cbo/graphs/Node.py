from cbo.graphs.StructuralEquations import StringEquation
import numpy as np


class Node:
    """
    A class representing a node in the causal graph.
    """

    def __init__(
        self, name, equation, parents_name=(), children_name=(), fixed_cost=0, variable_cost=False,
        min_intervention=None, max_intervention=None, is_reward_var=False
    ):
        """
        Initialise the node of a Graph
        :param name: the name of the node
        :param parents_name: the names of the parents nodes
        :param children_name: the names of the children node
        :param equation: the structural equation to use
        :param fixed_cost: the fixed cost
        :param variable_cost: A boolean indicating whether the cost is fixed or variable
        :param min_intervention: Minimum value of the intervention
        :param max_intervention: Maximum value of the intervention
        :param is_reward_var: whether the node corresponds to a reward variable
        """
        self._name = name
        self.parents_name = parents_name
        self.children_name = children_name
        self._equation = equation
        self._fixed_cost = fixed_cost
        self._variable_cost = variable_cost
        self._min_intervention = min_intervention
        self._max_intervention = max_intervention
        self.value = None
        self.is_reward_var = is_reward_var

    @property
    def name(self):
        return self._name

    @property
    def min_intervention(self):
        return self._min_intervention

    @property
    def max_intervention(self):
        return self._max_intervention

    def total_cost(self, interventions):
        """
        Compute the total cost of the interventions on the node
        :param interventions: the intervention values
        :return: the computed cost
        """
        cost = self._fixed_cost
        if self._variable_cost:
            cost += np.sum(np.abs(interventions))
        return cost

    def structural_equation(self, parents):
        """
        Update the value of the node during sampling
        """
        if isinstance(self._equation, StringEquation):
            # We allow named parameters here so that the lambdas can use the node names as variable names
            parent_values = {p.name: p.value for p in parents}
            self.value = np.float64(self._equation.predict(**parent_values))
        else:
            parent_values = np.array([[p.value for p in parents]])
            self.value = np.float64(self._equation.predict(parent_values))

    def fit_equation(self, node_measurement=None, parents_measurements=None):
        """
        Initially fit the structural equation if needed
        :param node_measurement: true measurements of the node
        :param parents_measurements: true measurements of its parents if they exist
        """
        if parents_measurements is None:
            self._equation.fit(node_measurement)
        else:
            # if both measurements are None, fit has no effect
            self._equation.fit(parents_measurements, node_measurement)

    def is_reward(self):
        """
        Check whether the node correspond to a reward variable
        :return: true, if the node correspond to a reward variable, false otherwise
        """
        return self.is_reward_var
