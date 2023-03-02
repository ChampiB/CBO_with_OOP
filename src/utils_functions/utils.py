from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from .cost_functions import *
from .causal_acquisition_functions import CausalExpectedImprovement
from .causal_optimizer import CausalGradientAcquisitionOptimizer


def find_current_global(current_y, dict_interventions, task):
    # This function finds the optimal value and variable at every iteration
    dict_values = {}
    for j in range(len(dict_interventions)):
        print(dict_interventions[j])
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



