from functools import partial
from src.graphs import Graph
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from .ToyGraph_DoFunctions import *
import sys
sys.path.append("../..")


class ToyGraph(Graph):
    """
    An instance of the class graph giving the graph structure in the toy example 
    
    Parameters
    ----------
    """

    def __init__(self, observational_samples):

        # Call the parent constructor
        super().__init__(['X', 'Z'])

        self.X = np.asarray(observational_samples['X'])[:, np.newaxis]
        self.Y = np.asarray(observational_samples['Y'])[:, np.newaxis]
        self.Z = np.asarray(observational_samples['Z'])[:, np.newaxis]

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

    def fit_all_gaussian_processes(self):

        num_features = self.Z.shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1., variance=1.)
        gp_Y = GPRegression(X=self.Z, Y=self.Y, kernel=kernel, noise_var=1.)
        gp_Y.optimize()

        num_features = self.X.shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1., variance=1.)
        gp_Z = GPRegression(X=self.X, Y=self.Z, kernel=kernel)
        gp_Z.optimize()

        return OrderedDict([
            ('Y', gp_Y),
            ('Z', gp_Z),
            ('X', [])
        ])

    def fit_all_gaussian_processes(self, observational_samples):
        X = np.asarray(observational_samples['X'])[:, np.newaxis]
        Z = np.asarray(observational_samples['Z'])[:, np.newaxis]
        Y = np.asarray(observational_samples['Y'])[:, np.newaxis]

        num_features = Z.shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1., variance=1.)
        gp_Y = GPRegression(X=Z, Y=Y, kernel=kernel, noise_var=1.)
        gp_Y.optimize()

        num_features = X.shape[1]
        kernel = RBF(num_features, ARD=False, lengthscale=1., variance=1.)
        gp_Z = GPRegression(X=X, Y=Z, kernel=kernel)
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
