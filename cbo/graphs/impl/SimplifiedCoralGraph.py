from functools import partial
from collections import OrderedDict
import scipy
from sklearn.linear_model import LinearRegression
import sklearn.mixture
from cbo.graphs import GraphInterface
import copy
import numpy as np
#import sys
#sys.path.append("../../..")
from cbo.utils_functions.utils import fit_gaussian_process


class SimplifiedCoralGraph(GraphInterface):
    """
    An instance of the class graph giving the graph structure in the Coral reef example 
    
    Parameters
    ----------
    """

    def __init__(self, measurements, true_measurements):

        # Call the parent constructor
        super().__init__(['N', 'O', 'C', 'T', 'D'])

        # The variable names
        self.var_names = ['Y', 'N', 'CO', 'T', 'D', 'P', 'O', 'S', 'L', 'TE', 'C']

        # The measurements made by the agents
        self.measurements = {
            var_name: np.asarray(measurements[var_name])[:, np.newaxis] for var_name in self.var_names
        }

        # The ground truth measurements
        self.true_measurements = {
            var_name: np.asarray(true_measurements[var_name])[:, np.newaxis] for var_name in self.var_names
        }

        self.fit_dependencies = [
            ["N"],
            ["O", "S", "T", "D", "TE"],
            ["C", "N", "L", "TE"],
            ["T", "S"],
            ["D", "S"],
            ["N", "O", "S", "T", "D", "TE"],
            ["N", "T", "S"],
            ["N", "D", "S"],
            ["O", "C", "N", "L", "TE", "S", "T", "D"],
            ["T", "C", "S", "TE", "L", "N"],
            ["T", "D", "S"],
            ["C", "D", "S", "TE", "L", "N"],
            ["N", "C", "T", "S", "N", "L", "TE"],
            ["N", "T", "D", "S"],
            ["C", "T", "D", "S", "N", "L", "TE"]
        ]

        # The dependencies between the variables
        self.var_dependencies = {
            "Y": ["L", "N", "P", "O", "C", "CO", "TE"],
            "P": ["S", "T", "D", "TE"],
            "O": ["S", "T", "D", "TE"],
            "CO": ["S", "T", "D", "TE"],
            "T": ["S"],
            "D": ["S"],
            "C": ["N", "L", "TE"],
            "S": ["TE"],
            "TE": ["L"],
        }

        self.fit_parameters = [
            [1., 1., 10., False],
            [1., 1., 1., True],
            [1., 1., 1., True],
            [1., 1., 1., True],
            [1., 1., 10., True],
            [1., 1., 1., False],
            [1., 1., 1., False],
            [1., 1., 1., False],
            [1., 1., 1., False],
            [1., 1., 1., False],
            [1., 1., 1., False],
            [1., 1., 1., False],
            [1., 1., 1., False],
            [1., 1., 1., False],
            [1., 1., 1., False]
        ]

        # Creating linear regression models and fit them on the available data.
        self.regressions = {}
        for var_name, dependencies in self.var_dependencies.items():
            inputs = np.hstack([self.true_measurements[dependency] for dependency in dependencies])
            self.regressions[var_name] = LinearRegression().fit(inputs, self.true_measurements[var_name])

        # Define distributions for the exogenous variables
        params_list = scipy.stats.gamma.fit(self.true_measurements["L"])
        self.dist_Light = scipy.stats.gamma(a=params_list[0], loc=params_list[1], scale=params_list[2])

        mixture = sklearn.mixture.GaussianMixture(n_components=3)
        mixture.fit(self.true_measurements["N"])
        self.dist_Nutrients_PC1 = mixture

    def define_sem(self):

        def fN(epsilon, **kwargs):
            return self.dist_Nutrients_PC1.sample(1)[0][0][0]

        def fL(epsilon, **kwargs):
            return self.dist_Light.rvs(1)[0]

        def fTE(epsilon, L, **kwargs):
            X = np.ones((1, 1)) * L
            return np.float64(self.regressions["TE"].predict(X))

        def fC(epsilon, N, L, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((N, L, TE))
            return np.float64(self.regressions["C"].predict(X))

        def fS(epsilon, TE, **kwargs):
            X = np.ones((1, 1)) * TE
            return np.float64(self.regressions["S"].predict(X))

        def fT(epsilon, S, **kwargs):
            X = np.ones((1, 1)) * S
            return np.float64(self.regressions["T"].predict(X))

        def fD(epsilon, S, **kwargs):
            X = np.ones((1, 1)) * S
            return np.float64(self.regressions["D"].predict(X))

        def fP(epsilon, S, T, D, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((S, T, D, TE))
            return np.float64(self.regressions["P"].predict(X))

        def fO(epsilon, S, T, D, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((S, T, D, TE))
            return np.float64(self.regressions["O"].predict(X))

        def fCO(epsilon, S, T, D, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((S, T, D, TE))
            return np.float64(self.regressions["CO"].predict(X))

        def fY(epsilon, L, N, P, O, C, CO, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((L, N, P, O, C, CO, TE))
            return np.float64(self.regressions["Y"].predict(X))

        graph = OrderedDict([
            ('N', fN),
            ('L', fL),
            ('TE', fTE),
            ('C', fC),
            ('S', fS),
            ('T', fT),
            ('D', fD),
            ('P', fP),
            ('O', fO),
            ('CO', fCO),
            ('Y', fY)
        ])

        return graph

    @staticmethod
    def get_exploration_set(set_name):
        MIS_1 = [['N'], ['O'], ['C'], ['T'], ['D']]
        MIS_2 = [['N', 'O'], ['N', 'C'], ['N', 'T'], ['N', 'D'], ['O', 'C'], ['O', 'T'], ['O', 'D'], ['T', 'C'],
                 ['T', 'D'], ['C', 'D']]
        MIS_3 = [
            ['N', 'O', 'C'], ['N', 'O', 'T'], ['N', 'O', 'D'], ['N', 'C', 'T'], ['N', 'C', 'D'], ['N', 'T', 'D'],
            ['O', 'C', 'T'], ['O', 'C', 'D'], ['C', 'T', 'D'], ['O', 'T', 'D']
        ]
        MIS_4 = [
            ['N', 'O', 'C', 'T'], ['N', 'O', 'C', 'D'], ['N', 'O', 'T', 'D'], ['N', 'T', 'D', 'C'], ['T', 'D', 'C', 'O']
        ]
        MIS_5 = [['N', 'O', 'C', 'T', 'D']]

        MIS = MIS_1 + MIS_2 + MIS_3

        # To change
        POMIS = MIS

        return MIS if set_name == "MIS" else POMIS

    @staticmethod
    def get_interventional_ranges():
        return OrderedDict([
            ('N', [-2, 5]),
            ('O', [3, 4]),
            ('C', [0.3, 0.4]),
            ('T', [2300, 2400]),
            ('D', [2000, 2080])
        ])

    def fit_all_gaussian_processes(self, measurements=None):
        # If no measurements provided as input, use the measurements in self
        measurements = self.measurements if measurements is None else {
            var_name: np.asarray(measurements[var_name])[:, np.newaxis]
            for var_name, measurement in measurements.items()
        }

        # Retrieve the measurements associated to each variable in the graph
        var_measurements = copy.deepcopy(measurements)

        # For each variable in the graph, concatenate the measurements of all the variables it depends on
        xs = [
            np.hstack([var_measurements[var_name] for var_name in dependencies])
            for dependencies in self.fit_dependencies
        ]

        # For each variable in the graph, retrieve the measurements of the target variable Y
        outputs = [measurements["Y"]] * len(xs)

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
                ('N', partial(self.cost, 1, False)),
                ('O', partial(self.cost, 1, False)),
                ('C', partial(self.cost, 1, False)),
                ('T', partial(self.cost, 1, False)),
                ('D', partial(self.cost, 1, False)),
            ])

        if type_cost == 2:
            return OrderedDict([
                ('N', partial(self.cost, 1, False)),
                ('O', partial(self.cost, 10, False)),
                ('C', partial(self.cost, 2, False)),
                ('T', partial(self.cost, 5, False)),
                ('D', partial(self.cost, 20, False)),
            ])

        if type_cost == 3:
            return OrderedDict([
                ('N', partial(self.cost, 1, True)),
                ('O', partial(self.cost, 10, True)),
                ('C', partial(self.cost, 2, True)),
                ('T', partial(self.cost, 5, True)),
                ('D', partial(self.cost, 20, True)),
            ])

        if type_cost == 4:
            return OrderedDict([
                ('N', partial(self.cost, 1, True)),
                ('O', partial(self.cost, 1, True)),
                ('C', partial(self.cost, 1, True)),
                ('T', partial(self.cost, 1, True)),
                ('D', partial(self.cost, 1, True)),
            ])

        raise RuntimeError(f"[ERROR] Invalid cost type: {type_cost}")
