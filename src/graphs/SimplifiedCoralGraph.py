from functools import partial
from collections import OrderedDict
import scipy
from sklearn.linear_model import LinearRegression
import sklearn.mixture
from src.graphs import Graph
from src.utils_functions import fit_gaussian_process
from .SimplifiedCoralGraph_DoFunctions import *
import sys
sys.path.append("../..")


class SimplifiedCoralGraph(Graph):
    """
    An instance of the class graph giving the graph structure in the Coral reef example 
    
    Parameters
    ----------
    """

    def __init__(self, observational_samples, true_observational_samples):

        # Call the parent constructor
        super().__init__(['N', 'O', 'C', 'T', 'D'])

        self.Y = np.asarray(observational_samples['Y'])[:, np.newaxis]
        self.N = np.asarray(observational_samples['N'])[:, np.newaxis]
        self.CO = np.asarray(observational_samples['CO'])[:, np.newaxis]
        self.T = np.asarray(observational_samples['T'])[:, np.newaxis]
        self.D = np.asarray(observational_samples['D'])[:, np.newaxis]
        self.P = np.asarray(observational_samples['P'])[:, np.newaxis]
        self.O = np.asarray(observational_samples['O'])[:, np.newaxis]
        self.S = np.asarray(observational_samples['S'])[:, np.newaxis]
        self.L = np.asarray(observational_samples['L'])[:, np.newaxis]
        self.TE = np.asarray(observational_samples['TE'])[:, np.newaxis]
        self.C = np.asarray(observational_samples['C'])[:, np.newaxis]

        true_Y = np.asarray(true_observational_samples['Y'])[:, np.newaxis]
        true_N = np.asarray(true_observational_samples['N'])[:, np.newaxis]
        true_CO = np.asarray(true_observational_samples['CO'])[:, np.newaxis]
        true_T = np.asarray(true_observational_samples['T'])[:, np.newaxis]
        true_D = np.asarray(true_observational_samples['D'])[:, np.newaxis]
        true_P = np.asarray(true_observational_samples['P'])[:, np.newaxis]
        true_O = np.asarray(true_observational_samples['O'])[:, np.newaxis]
        true_S = np.asarray(true_observational_samples['S'])[:, np.newaxis]
        true_L = np.asarray(true_observational_samples['L'])[:, np.newaxis]
        true_TE = np.asarray(true_observational_samples['TE'])[:, np.newaxis]
        true_C = np.asarray(true_observational_samples['C'])[:, np.newaxis]

        self.reg_Y = LinearRegression().fit(np.hstack((true_L, true_N, true_P, true_O, true_C, true_CO, true_TE)), true_Y)
        self.reg_P = LinearRegression().fit(np.hstack((true_S, true_T, true_D, true_TE)), true_P)
        self.reg_O = LinearRegression().fit(np.hstack((true_S, true_T, true_D, true_TE)), true_O)
        self.reg_CO = LinearRegression().fit(np.hstack((true_S, true_T, true_D, true_TE)), true_CO)
        self.reg_T = LinearRegression().fit(true_S, true_T)
        self.reg_D = LinearRegression().fit(true_S, true_D)
        self.reg_C = LinearRegression().fit(np.hstack((true_N, true_L, true_TE)), true_C)
        self.reg_S = LinearRegression().fit(true_TE, true_S)
        self.reg_TE = LinearRegression().fit(true_L, true_TE)

        # Define distributions for the exogenous variables
        params_list = scipy.stats.gamma.fit(true_L)
        self.dist_Light = scipy.stats.gamma(a=params_list[0], loc=params_list[1], scale=params_list[2])

        mixture = sklearn.mixture.GaussianMixture(n_components=3)
        mixture.fit(true_N)
        self.dist_Nutrients_PC1 = mixture

    def define_sem(self):

        def fN(epsilon, **kwargs):
            return self.dist_Nutrients_PC1.sample(1)[0][0][0]

        def fL(epsilon, **kwargs):
            return self.dist_Light.rvs(1)[0]

        def fTE(epsilon, L, **kwargs):
            X = np.ones((1, 1)) * L
            return np.float64(self.reg_TE.predict(X))

        def fC(epsilon, N, L, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((N, L, TE))
            return np.float64(self.reg_C.predict(X))

        def fS(epsilon, TE, **kwargs):
            X = np.ones((1, 1)) * TE
            return np.float64(self.reg_S.predict(X))

        def fT(epsilon, S, **kwargs):
            X = np.ones((1, 1)) * S
            return np.float64(self.reg_T.predict(X))

        def fD(epsilon, S, **kwargs):
            X = np.ones((1, 1)) * S
            return np.float64(self.reg_D.predict(X))

        def fP(epsilon, S, T, D, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((S, T, D, TE))
            return np.float64(self.reg_P.predict(X))

        def fO(epsilon, S, T, D, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((S, T, D, TE))
            return np.float64(self.reg_O.predict(X))

        def fCO(epsilon, S, T, D, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((S, T, D, TE))
            return np.float64(self.reg_CO.predict(X))

        def fY(epsilon, L, N, P, O, C, CO, TE, **kwargs):
            X = np.ones((1, 1)) * np.hstack((L, N, P, O, C, CO, TE))
            return np.float64(self.reg_Y.predict(X))

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

    def fit_all_gaussian_processes(self):
        functions = {}
        inputs_list = [self.N, np.hstack((self.O, self.S, self.T, self.D, self.TE)),
                       np.hstack((self.C, self.N, self.L, self.TE)), np.hstack((self.T, self.S)),
                       np.hstack((self.D, self.S)), np.hstack((self.N, self.O, self.S, self.T, self.D, self.TE)),
                       np.hstack((self.N, self.T, self.S)),
                       np.hstack((self.N, self.D, self.S)),
                       np.hstack((self.O, self.C, self.N, self.L, self.TE, self.S, self.T, self.D)),
                       np.hstack((self.T, self.C, self.S, self.TE, self.L, self.N)),
                       np.hstack((self.T, self.D, self.S)),
                       np.hstack((self.C, self.D, self.S, self.TE, self.L, self.N)),
                       np.hstack((self.N, self.C, self.T, self.S, self.N, self.L, self.TE)),
                       np.hstack((self.N, self.T, self.D, self.S)),
                       np.hstack((self.C, self.T, self.D, self.S, self.N, self.L, self.TE))]

        output_list = [self.Y, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y, self.Y,
                       self.Y, self.Y, self.Y]

        name_list = ['gp_N', 'gp_O_S_T_D_TE', 'gp_C_N_L_TE', 'gp_T_S', 'gp_D_S', 'gp_N_O_S_T_D_TE', 'gp_N_T_S',
                     'gp_N_D_S', 'gp_O_C_N_L_TE_S_T_D',
                     'gp_T_C_S_TE_L_N', 'gp_T_D_S', 'gp_C_D_S_TE_L_N', 'gp_N_C_T_S_N_L_TE', 'gp_N_T_D_S',
                     'gp_C_T_D_S_N_L_TE']

        parameter_list = [[1., 1., 10., False], [1., 1., 1., True], [1., 1., 1., True], [1., 1., 1., True],
                          [1., 1., 10., True],
                          [1., 1., 1., False], [1., 1., 1., False], [1., 1., 1., False], [1., 1., 1., False],
                          [1., 1., 1., False], [1., 1., 1., False], [1., 1., 1., False],
                          [1., 1., 1., False], [1., 1., 1., False], [1., 1., 1., False]]

        # Fit all conditional models
        for i in range(len(inputs_list)):
            functions[name_list[i]] = fit_gaussian_process(inputs_list[i], output_list[i], parameter_list[i])

        return functions

    def fit_all_gaussian_processes(self, observational_samples):
        Y = np.asarray(observational_samples['Y'])[:, np.newaxis]
        N = np.asarray(observational_samples['N'])[:, np.newaxis]
        CO = np.asarray(observational_samples['CO'])[:, np.newaxis]
        T = np.asarray(observational_samples['T'])[:, np.newaxis]
        D = np.asarray(observational_samples['D'])[:, np.newaxis]
        P = np.asarray(observational_samples['P'])[:, np.newaxis]
        O = np.asarray(observational_samples['O'])[:, np.newaxis]
        S = np.asarray(observational_samples['S'])[:, np.newaxis]
        L = np.asarray(observational_samples['L'])[:, np.newaxis]
        TE = np.asarray(observational_samples['TE'])[:, np.newaxis]
        C = np.asarray(observational_samples['C'])[:, np.newaxis]

        functions = {}
        inputs_list = [N, np.hstack((O, S, T, D, TE)), np.hstack((C, N, L, TE)), np.hstack((T, S)),
                       np.hstack((D, S)), np.hstack((N, O, S, T, D, TE)), np.hstack((N, T, S)),
                       np.hstack((N, D, S)), np.hstack((O, C, N, L, TE, S, T, D)),
                       np.hstack((T, C, S, TE, L, N)), np.hstack((T, D, S)),
                       np.hstack((C, D, S, TE, L, N)), np.hstack((N, C, T, S, N, L, TE)),
                       np.hstack((N, T, D, S)), np.hstack((C, T, D, S, N, L, TE))]

        output_list = [Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y, Y]

        name_list = ['gp_N', 'gp_O_S_T_D_TE', 'gp_C_N_L_TE', 'gp_T_S', 'gp_D_S', 'gp_N_O_S_T_D_TE', 'gp_N_T_S',
                     'gp_N_D_S', 'gp_O_C_N_L_TE_S_T_D',
                     'gp_T_C_S_TE_L_N', 'gp_T_D_S', 'gp_C_D_S_TE_L_N', 'gp_N_C_T_S_N_L_TE', 'gp_N_T_D_S',
                     'gp_C_T_D_S_N_L_TE']

        parameter_list = [[1., 1., 10., False], [1., 1., 1., True], [1., 1., 1., True], [1., 1., 1., True],
                          [1., 1., 10., True],
                          [1., 1., 1., False], [1., 1., 1., False], [1., 1., 1., False], [1., 1., 1., False],
                          [1., 1., 1., False], [1., 1., 1., False], [1., 1., 1., False],
                          [1., 1., 1., False], [1., 1., 1., False], [1., 1., 1., False]]

        # Fit all conditional models
        for i in range(len(inputs_list)):
            X = inputs_list[i]
            Y = output_list[i]
            functions[name_list[i]] = fit_gaussian_process(X, Y, parameter_list[i])

        return functions

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
