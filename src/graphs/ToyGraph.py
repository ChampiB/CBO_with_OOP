from graph import GraphStructure
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from .ToyGraph_DoFunctions import *
from .ToyGraph_CostFunctions import define_costs
import sys
sys.path.append("../..")


class ToyGraph(GraphStructure):
    """
    An instance of the class graph giving the graph structure in the toy example 
    
    Parameters
    ----------
    """

    def __init__(self, observational_samples):

        self.X = np.asarray(observational_samples['X'])[:,np.newaxis]
        self.Y = np.asarray(observational_samples['Y'])[:,np.newaxis]
        self.Z = np.asarray(observational_samples['Z'])[:,np.newaxis]
        self.do_functions = {'compute_do_X': compute_do_X, 'compute_do_Z': compute_do_Z, 'compute_do_XZ': compute_do_XZ}

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

    def get_sets(self):
        MIS = [['X'], ['Z']]
        POMIS = [['Z']]
        manipulative_variables = ['X', 'Z']
        return MIS, POMIS, manipulative_variables

    def get_set_BO(self):
        manipulative_variables = ['X', 'Z']
        return manipulative_variables

    def get_interventional_ranges(self):
        return OrderedDict([
            ('X', [-5, 5]),
            ('Z', [-5, 20]),
        ])

    def fit_all_gaussian_processes(self):
        functions = {}

        num_features = self.Z.shape[1]
        kernel = RBF(num_features, ARD = False, lengthscale=1., variance = 1.) 
        gp_Y = GPRegression(X = self.Z, Y = self.Y, kernel = kernel, noise_var= 1.)
        gp_Y.optimize()

        num_features = self.X.shape[1]
        kernel = RBF(num_features, ARD = False, lengthscale=1., variance =1.) 
        gp_Z = GPRegression(X = self.X, Y = self.Z, kernel = kernel)
        gp_Z.optimize()

        functions = OrderedDict ([
            ('Y', gp_Y),
            ('Z', gp_Z),
            ('X', [])
            ])

        return functions

    def fit_all_gaussian_processes(self, observational_samples):
        X = np.asarray(observational_samples['X'])[:,np.newaxis]
        Z = np.asarray(observational_samples['Z'])[:,np.newaxis]
        Y = np.asarray(observational_samples['Y'])[:,np.newaxis]

        functions = {}

        num_features = Z.shape[1]
        kernel = RBF(num_features, ARD = False, lengthscale=1., variance = 1.) 
        gp_Y = GPRegression(X = Z, Y = Y, kernel = kernel, noise_var= 1.)
        gp_Y.optimize()

        num_features = X.shape[1]
        kernel = RBF(num_features, ARD = False, lengthscale=1., variance =1.) 
        gp_Z = GPRegression(X = X, Y = Z, kernel = kernel)
        gp_Z.optimize()
        
        functions = OrderedDict ([
            ('Y', gp_Y),
            ('Z', gp_Z),
            ('X', [])
            ])

        return functions

    def get_cost_structure(self, type_cost):
        costs = define_costs(type_cost)
        return costs

    def get_do_function(self, intervention):
        name = self.get_function_name(intervention)
        return self.do_functions[name]
