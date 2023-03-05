from functools import partial
from cbo.graphs import GraphInterface
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from cbo.graphs.ToyGraph_DoFunctions import *
#import sys
#sys.path.append("../../..")


class ToyGraph(GraphInterface):
    """
    An instance of the class graph giving the graph structure in the toy example 
    
    Parameters
    ----------
    """

    def __init__(self, measurements):

        # Call the parent constructor
        super().__init__(['X', 'Z'])

        # The variable names
        self.var_names = ['X', 'Y', 'Z']

        # The measurements made by the agents
        self.measurements = {
            var_name: np.asarray(measurements[var_name])[:, np.newaxis] for var_name in self.var_names
        }

    def define_sem(self):

        def fx(epsilon, **kwargs):
            return epsilon[0]

        def fz(epsilon, X, **kwargs):
            return np.exp(-X) + epsilon[1]

        def fy(epsilon, Z, **kwargs):
            return np.cos(Z) - np.exp(-Z/20.) + epsilon[2]

        graph = OrderedDict([
          ('X', fx),
          ('Z', fz),
          ('Y', fy),
        ])

        return graph

    @staticmethod
    def get_exploration_set(set_name):
        MIS = [['X'], ['Z']]
        POMIS = [['Z']]
        return MIS if set_name == "MIS" else POMIS

    @staticmethod
    def get_interventional_ranges():
        return OrderedDict([
            ('X', [-5, 5]),
            ('Z', [-5, 20]),
        ])

    def fit_all_gaussian_processes(self, measurements=None):
        # If no measurements provided as input, use the measurements in self
        measurements = self.measurements if measurements is None else {
            var_name: np.asarray(measurements[var_name])[:, np.newaxis]
            for var_name, measurement in measurements.items()
        }

        num_features = measurements["Z"].shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1., variance=1.)
        gp_Y = GPRegression(X=measurements["Z"], Y=measurements["Y"], kernel=kernel, noise_var=1.)
        gp_Y.optimize()

        num_features = measurements["X"].shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1., variance=1.)
        gp_Z = GPRegression(X=measurements["X"], Y=measurements["Z"], kernel=kernel)
        gp_Z.optimize()

        return OrderedDict([
            ('Y', gp_Y),
            ('Z', gp_Z),
            ('X', [])
        ])

    def get_cost_structure(self, type_cost):

        if type_cost == 1:
            return OrderedDict([
                ('X', partial(self.cost, 1, False)),
                ('Z', partial(self.cost, 1, False)),
            ])

        if type_cost == 2:
            return OrderedDict([
                ('X', partial(self.cost, 1, False)),
                ('Z', partial(self.cost, 10, False)),
            ])

        if type_cost == 3:
            return OrderedDict([
                ('X', partial(self.cost, 1, True)),
                ('Z', partial(self.cost, 10, True)),
            ])

        if type_cost == 4:
            return OrderedDict([
                ('X', partial(self.cost, 1, True)),
                ('Z', partial(self.cost, 1, True)),
            ])

        raise RuntimeError(f"[ERROR] Invalid cost type: {type_cost}")
