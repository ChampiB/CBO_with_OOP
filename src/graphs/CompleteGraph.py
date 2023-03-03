from functools import partial
from collections import OrderedDict
from src.graphs import Graph
from src.utils_functions import fit_gaussian_process
from .CompleteGraph_DoFunctions import *
import sys
sys.path.append("../..")


class CompleteGraph(Graph):
    """
    An instance of the class graph giving the graph structure in the synthetic example 
    
    Parameters
    ----------
    """

    def __init__(self, observational_samples):

        # Call the parent constructor
        super().__init__(['B', 'D', 'E'])

        self.A = np.asarray(observational_samples['A'])[:, np.newaxis]
        self.B = np.asarray(observational_samples['B'])[:, np.newaxis]
        self.C = np.asarray(observational_samples['C'])[:, np.newaxis]
        self.D = np.asarray(observational_samples['D'])[:, np.newaxis]
        self.E = np.asarray(observational_samples['E'])[:, np.newaxis]
        self.F = np.asarray(observational_samples['F'])[:, np.newaxis]
        self.Y = np.asarray(observational_samples['Y'])[:, np.newaxis]

    def define_sem(self):

        def fU1(epsilon, **kwargs):
          return epsilon[0]

        def fU2(epsilon, **kwargs):
          return epsilon[1]

        def fF(epsilon, **kwargs):
          return epsilon[8]

        def fA(epsilon, U1, F, **kwargs):
          return F**2 + U1 + epsilon[2]

        def fB(epsilon, U2, **kwargs):
          return U2 + epsilon[3]

        def fC(epsilon, B, **kwargs):
          return np.exp(-B) + epsilon[4]

        def fD(epsilon, C, **kwargs):
          return np.exp(-C)/10. + epsilon[5]

        def fE(epsilon, A, C, **kwargs):
          return np.cos(A) + C/10. + epsilon[6]

        def fY(epsilon, D, E, U1, U2, **kwargs):
          return np.cos(D) - D/5. + np.sin(E) - E/4. + U1 + np.exp(-U2) + epsilon[7]

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

    def get_interventional_ranges(self):
        min_intervention_e = -6
        max_intervention_e = 3

        min_intervention_b = -5
        max_intervention_b = 4

        min_intervention_d = -5
        max_intervention_d = 5

        min_intervention_f = -4
        max_intervention_f = 4

        dict_ranges = OrderedDict ([
          ('E', [min_intervention_e, max_intervention_e]),
          ('B', [min_intervention_b, max_intervention_b]),
          ('D', [min_intervention_d, max_intervention_d]),
          ('F', [min_intervention_f, max_intervention_f])
        ])
        return dict_ranges

    def fit_all_gaussian_processes(self):
        functions = {}
        inputs_list = [self.B, self.F, np.hstack((self.D,self.C)), np.hstack((self.B,self.C)), np.hstack((self.A,self.C,self.E)), np.hstack((self.B,self.C,self.D)), 
                    np.hstack((self.D,self.E,self.C,self.A)),np.hstack((self.B,self.E,self.C,self.A)), np.hstack((self.A,self.B,self.C,self.D,self.E)), 
                    np.hstack((self.A,self.B,self.C,self.D,self.E, self.F))]
        output_list = [self.C, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y]
        name_list = ['gp_C', 'gp_A', 'gp_D_C', 'gp_B_C', 'gp_A_C_E', 'gp_B_C_D', 'gp_D_E_C_A', 'gp_B_E_C_A', 'gp_A_B_C_D_E', 'gp_A_B_C_D_E_F']
        parameter_list = [[1.,1.,0.0001, False], [1.,1.,10., False], [1.,1.,1., False], [1.,1.,1., False], [1.,1.,10., False], 
                            [1.,1.,1., False], [1.,1.,10., False], [1.,1.,10., False], [1.,1.,10., False],[1.,1.,10., False]]

        # Fit all conditional models
        for i in range(len(inputs_list)):
            X = inputs_list[i]
            Y = output_list[i]
            functions[name_list[i]] = fit_gaussian_process(X, Y, parameter_list[i])

        return functions

    def fit_all_gaussian_processes(self, observational_samples):
        A = np.asarray(observational_samples['A'])[:,np.newaxis]
        B = np.asarray(observational_samples['B'])[:,np.newaxis]
        C = np.asarray(observational_samples['C'])[:,np.newaxis]
        D = np.asarray(observational_samples['D'])[:,np.newaxis]
        E = np.asarray(observational_samples['E'])[:,np.newaxis]
        F = np.asarray(observational_samples['F'])[:,np.newaxis]
        Y = np.asarray(observational_samples['Y'])[:,np.newaxis]

        functions = {}
        inputs_list = [B, np.hstack((A,C,E)), np.hstack((D,C)), np.hstack((B,C)), np.hstack((B,C,D)), 
                    np.hstack((D,E,C,A)),np.hstack((B,E,C,A))]
        output_list = [C, Y, Y, Y, Y, Y, Y, Y]
        name_list = ['gp_C', 'gp_A_C_E', 'gp_D_C', 'gp_B_C', 'gp_B_C_D', 'gp_D_E_C_A', 'gp_B_E_C_A']
        parameter_list = [[1.,1.,10., False], [1.,1.,10., False], [1.,1.,1., False], [1.,1.,10., False], [1.,1.,10., False], [1.,1.,10., False], [1.,1.,10., False]]

        ## Fit all conditional models
        for i in range(len(inputs_list)):
            X = inputs_list[i]
            Y = output_list[i]
            functions[name_list[i]] = fit_gaussian_process(X, Y, parameter_list[i])
  
        return functions

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
