import os.path
import matplotlib.pyplot as plt
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
import numpy as np

from cbo.utils_functions.causal_acquisition_functions import CausalExpectedImprovement
from cbo.utils_functions.causal_optimizer import CausalGradientAcquisitionOptimizer
from cbo.utils_functions.cost_functions import Cost


def find_current_global(current_y, dict_interventions, task):
    # This function finds the optimal value and variable at every iteration
    dict_values = {}
    for j in range(len(dict_interventions)):
        dict_values[dict_interventions[j]] = []

    for variable, value in current_y.items():
        if len(value) > 0:
            if task == 'min':
                dict_values[variable] = np.min(current_y[variable])
            else:
                dict_values[variable] = np.max(current_y[variable])
    if task == 'min':        
        opt_variable = min(dict_values, key=dict_values.get)
    else:
        opt_variable = max(dict_values, key=dict_values.get)
    
    opt_value = dict_values[opt_variable]
    return opt_value


def find_next_y_point(space, model, current_global_best, evaluated_set, costs_functions, task='min'):
    # This function optimises the acquisition function and return the next point together with the
    # corresponding y value for the acquisition function
    cost_acquisition = Cost(costs_functions, evaluated_set)
    optimizer = CausalGradientAcquisitionOptimizer(space)
    acquisition = CausalExpectedImprovement(current_global_best, task, model) / cost_acquisition
    x_new, _ = optimizer.optimize(acquisition)
    y_acquisition = acquisition.evaluate(x_new)  
    return y_acquisition, x_new    


def fit_gaussian_process(x, y, parameter_list):
    kernel = RBF(x.shape[1], ARD=parameter_list[3], lengthscale=parameter_list[0], variance=parameter_list[1])
    gp = GPRegression(X=x, Y=y, kernel=kernel, noise_var=parameter_list[2])
    gp.likelihood.variance.fix(1e-2)
    gp.optimize()
    return gp


def is_valid_path(path):
    """ Check if a path exists

    :param path: the path to check (can be None)
    :return: True if the path is valid false otherwise
    """
    return path is not None and os.path.exists(path)


def save_figure(out_fname, dpi=300, tight=True):
    """Save a matplotlib figure in an `out_fname` file.

    :param out_fname:  Name of the file used to save the figure.
    :param dpi: Number of dpi, Default 300.
    :param tight: If True, use plt.tight_layout() before saving. Default True.
    :return: None
    """
    if tight is True:
        plt.tight_layout()
    plt.savefig(out_fname, dpi=dpi, transparent=True)
    plt.clf()
    plt.cla()
    plt.close()