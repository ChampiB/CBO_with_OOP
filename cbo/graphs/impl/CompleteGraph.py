from functools import partial
from collections import OrderedDict
from cbo.graphs import GraphInterface
import numpy as np
#import sys
#sys.path.append("../../..")
from cbo.utils_functions.utils import fit_gaussian_process


class CompleteGraph(GraphInterface):
    """
    An instance of the class graph giving the graph structure in the synthetic example
    
    Parameters
    ----------
    """

    def __init__(self, measurements):

        # Call the parent constructor
        super().__init__(['B', 'D', 'E'])

        # The variable names
        self.var_names = ['A', 'B', 'C', 'D', 'E', 'F', 'Y']

        # The measurements made by the agents
        self.measurements = {
            var_name: np.asarray(measurements[var_name])[:, np.newaxis] for var_name in self.var_names
        }

        self.fit_dependencies = [
            ["B"],
            ["F"],
            ["D", "C"],
            ["B", "C"],
            ["A", "C", "E"],
            ["B", "C", "D"],
            ["D", "E", "C", "A"],
            ["B", "E", "C", "A"],
            ["A", "B", "C", "D", "E"],
            ["A", "B", "C", "D", "E", "F"],
        ]

        self.fit_parameters = [
            [1., 1., 0.0001, False],
            [1., 1., 10., False],
            [1., 1., 1., False],
            [1., 1., 1., False],
            [1., 1., 10., False],
            [1., 1., 1., False],
            [1., 1., 10., False],
            [1., 1., 10., False],
            [1., 1., 10., False],
            [1., 1., 10., False]
        ]

    def define_sem(self):

        def fU1(epsilon, **kwargs):
            return epsilon[0]

        def fU2(epsilon, **kwargs):
            return epsilon[1]

        def fF(epsilon, **kwargs):
            return epsilon[8]

        def fA(epsilon, U1, F, **kwargs):
            return F ** 2 + U1 + epsilon[2]

        def fB(epsilon, U2, **kwargs):
            return U2 + epsilon[3]

        def fC(epsilon, B, **kwargs):
            return np.exp(-B) + epsilon[4]

        def fD(epsilon, C, **kwargs):
            return np.exp(-C) / 10. + epsilon[5]

        def fE(epsilon, A, C, **kwargs):
            return np.cos(A) + C / 10. + epsilon[6]

        def fY(epsilon, D, E, U1, U2, **kwargs):
            return np.cos(D) - D / 5. + np.sin(E) - E / 4. + U1 + np.exp(-U2) + epsilon[7]

        graph = OrderedDict([
            ('U1', fU1),
            ('U2', fU2),
            ('F', fF),
            ('A', fA),
            ('B', fB),
            ('C', fC),
            ('D', fD),
            ('E', fE),
            ('Y', fY),
        ])
        return graph

    @staticmethod
    def get_exploration_set(set_name):
        MIS = [['B'], ['D'], ['E'], ['B', 'D'], ['B', 'E'], ['D', 'E']]
        POMIS = [['B'], ['D'], ['E'], ['B', 'D'], ['D', 'E']]
        return MIS if set_name == "MIS" else POMIS

    @staticmethod
    def get_interventional_ranges():
        return OrderedDict([
            ('E', [-6, 3]),
            ('B', [-5, 4]),
            ('D', [-5, 5]),
            ('F', [-4, 4])
        ])

    def fit_all_gaussian_processes(self, measurements=None):
        # If no measurements provided as input, use the measurements in self
        measurements = self.measurements if measurements is None else {
            var_name: np.asarray(measurements[var_name])[:, np.newaxis]
            for var_name, measurement in measurements.items()
        }

        # For each variable in the graph, concatenate the measurements of all the variables it depends on
        xs = [
            np.hstack([self.measurements[var_name] for var_name in dependencies])
            for dependencies in self.fit_dependencies
        ]

        # For each variable in the graph, retrieve the measurements of the target variable Y
        outputs = [measurements["C"]] + [measurements["Y"]] * (len(xs) - 1)  # TODO is this an error? or should C be C?

        # Create the name of the Gaussian process corresponding to each variable in the graph
        names = ["gp_" + "_".join(dependencies) for dependencies in self.fit_dependencies]

        # Fit all conditional Gaussian processes
        return {
            name: fit_gaussian_process(x, output, parameter)
            for name, x, output, parameter in zip(names, xs, outputs, self.fit_parameters)
        }

    def get_cost_structure(self, type_cost):

        if type_cost == 1:
            return OrderedDict([
                ('A', partial(self.cost, 1, False)),
                ('B', partial(self.cost, 1, False)),
                ('C', partial(self.cost, 1, False)),
                ('D', partial(self.cost, 1, False)),
                ('E', partial(self.cost, 1, False)),
                ('F', partial(self.cost, 1, False)),
            ])

        if type_cost == 2:
            return OrderedDict([
                ('A', partial(self.cost, 1, False)),
                ('B', partial(self.cost, 10, False)),
                ('C', partial(self.cost, 2, False)),
                ('D', partial(self.cost, 5, False)),
                ('E', partial(self.cost, 20, False)),
                ('F', partial(self.cost, 3, False)),
            ])

        if type_cost == 3:
            return OrderedDict([
                ('A', partial(self.cost, 1, True)),
                ('B', partial(self.cost, 10, True)),
                ('C', partial(self.cost, 2, True)),
                ('D', partial(self.cost, 5, True)),
                ('E', partial(self.cost, 20, True)),
                ('F', partial(self.cost, 3, True)),
            ])

        if type_cost == 4:
            return OrderedDict([
                ('A', partial(self.cost, 1, True)),
                ('B', partial(self.cost, 1, True)),
                ('C', partial(self.cost, 1, True)),
                ('D', partial(self.cost, 1, True)),
                ('E', partial(self.cost, 1, True)),
                ('F', partial(self.cost, 1, True)),
            ])

        raise RuntimeError(f"[ERROR] Invalid cost type: {type_cost}")
