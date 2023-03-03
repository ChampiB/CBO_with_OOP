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
        self._parents = []
        self._children = []
        self._value = None
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

    @property
    def parents(self):
        return self._parents

    @parents.setter
    def parents(self, parents):
        self.parents = parents

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, children):
        self.children = children

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def total_cost(self, interventions):
        return self._fixed_cost + self._variable_cost(interventions)

    def structural_equation(self):
        parent_values = [p.value for p in self._parents]
        self.value = self._equation(parent_values)
