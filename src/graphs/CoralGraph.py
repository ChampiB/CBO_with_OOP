import copy
from collections import OrderedDict
from scipy.stats import gamma
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from src.graphs import GraphStructure
from src.utils_functions import fit_gaussian_process
from .CoralGraph_CostFunctions import define_costs
import sys
sys.path.append("../..")


class CoralGraph(GraphStructure):
    """
    An instance of the class graph giving the graph structure in the Coral reef example

    Parameters
    ----------
    """

    def __init__(self, measurements, true_measurements):

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

        # The dependencies between the variables when acting upon some of them
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
        a, loc, scale = gamma.fit(self.true_measurements['L'])
        self.dist_Light = gamma(a=a, loc=loc, scale=scale)

        self.dist_nutrients_pc1 = GaussianMixture(n_components=3)
        self.dist_nutrients_pc1.fit(self.true_measurements['N'])

    def define_sem(self):

        def f_n(epsilon, **kwargs):
            return self.dist_nutrients_pc1.sample(1)[0][0][0]

        def f_l(epsilon, **kwargs):
            return self.dist_Light.rvs(1)[0]

        def f_te(epsilon, L, **kwargs):
            X = np.ones((1, 1)) * L
            return np.float64(self.regressions["TE"].predict(X))

        def f_c(epsilon, N, L, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((N, L, TE))
            return np.float64(self.regressions["C"].predict(X))

        def f_s(epsilon, TE, **kwargs):
            X = np.ones((1, 1)) * TE
            return np.float64(self.regressions["S"].predict(X))

        def f_t(epsilon, S, **kwargs):
            X = np.ones((1, 1)) * S
            return np.float64(self.regressions["T"].predict(X))

        def f_d(epsilon, S, **kwargs):
            X = np.ones((1, 1)) * S
            return np.float64(self.regressions["D"].predict(X))

        def f_p(epsilon, S, T, D, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((S, T, D, TE))
            return np.float64(self.regressions["P"].predict(X))

        def f_o(epsilon, S, T, D, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((S, T, D, TE))
            return np.float64(self.regressions["O"].predict(X))

        def f_co(epsilon, S, T, D, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((S, T, D, TE))
            return np.float64(self.regressions["CO"].predict(X))

        def f_y(epsilon, L, N, P, O, C, CO, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((L, N, P, O, C, CO, TE))
            return np.float64(self.regressions["Y"].predict(X))

        return OrderedDict([
            ('N', f_n),
            ('L', f_l),
            ('TE', f_te),
            ('C', f_c),
            ('S', f_s),
            ('T', f_t),
            ('D', f_d),
            ('P', f_p),
            ('O', f_o),
            ('CO', f_co),
            ('Y', f_y)
        ])

    @staticmethod
    def get_exploration_set(set_name):
        MIS_1 = [['N'], ['O'], ['C'], ['T'], ['D']]
        MIS_2 = [['N', 'O'], ['N', 'C'], ['N', 'T'], ['N', 'D'], ['O', 'C'], ['O', 'T'], ['O', 'D'], ['T', 'C'], ['T', 'D'], ['C', 'D']]
        MIS_3 = [['N', 'O', 'C'], ['N', 'O', 'T'], ['N', 'O', 'D'], ['N', 'C', 'T'], ['N', 'C', 'D'], ['N', 'T', 'D'], ['O', 'C', 'T'], ['O', 'C', 'D'], ['C', 'T', 'D'], ['O', 'T', 'D']]
        MIS_4 = [['N', 'O', 'C', 'T'], ['N', 'O', 'C', 'D'], ['N', 'O', 'T', 'D'], ['N', 'T', 'D', 'C'], ['T', 'D', 'C', 'O']]
        MIS_5 = [['N', 'O', 'C', 'T', 'D']]

        MIS = MIS_1 + MIS_2 + MIS_3

        # To change
        POMIS = MIS

        return MIS if set_name == "MIS" else POMIS

    @staticmethod
    def get_manipulative_variables():
        return ['N', 'O', 'C', 'T', 'D']

    @staticmethod
    def get_interventional_ranges():
        return OrderedDict([
          ('N', [-2, 5]),
          ('O', [2, 4]),
          ('C', [0, 1]),
          ('T', [2450, 2500]),
          ('D', [1950, 1965])
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

    @staticmethod
    def get_cost_structure(type_cost):
        return define_costs(type_cost)
