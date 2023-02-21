import numpy as np


def get_do_function_name(intervention_variables):
    return 'compute_do_' + "".join(intervention_variables)


# Given a do function, this function is computing the mean and variance functions needed for the Causal prior
def mean_var_do_functions(do_effects_function, observational_samples, functions):
    xi_dict_mean = {}

    def mean_function_do(x):
        num_interventions = x.shape[0]
        mean_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            xi_str = str(x[i])
            if xi_str in xi_dict_mean:
                mean_do[i] = xi_dict_mean[xi_str]
            else:
                mean_do[i], _ = do_effects_function(observational_samples, functions, x[i])
                xi_dict_mean[xi_str] = mean_do[i]
        return np.float64(mean_do)
    
    xi_dict_var = {}

    def var_function_do(x):
        num_interventions = x.shape[0]
        var_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            xi_str = str(x[i])
            if xi_str in xi_dict_var:
                var_do[i] = xi_dict_var[xi_str]
            else:
                _, var_do[i] = do_effects_function(observational_samples, functions, x[i])
                xi_dict_var[xi_str] = var_do[i]
        return np.float64(var_do)

    return mean_function_do, var_function_do


def update_mean_fun(graph, functions, variables, observational_samples, xi_dict_mean):

    do_functions = graph.get_all_do()
    function_name = get_do_function_name(variables)

    def mean_function_do(x):
        # Get the number of interventions
        n_interventions = x.shape[0]

        # Compute mean do?
        xi_mean = xi_dict_mean[variables]
        mean_do = np.zeros((n_interventions, 1))
        for i in range(n_interventions):
            xi_str = str(x[i])
            if xi_str in xi_mean:
                mean_do[i] = xi_mean[xi_str]
            else:
                mean_do[i] = do_functions[function_name](observational_samples, functions, value=x[i])[0]
                xi_mean[xi_str] = mean_do[i]
        return np.float64(mean_do)

    return mean_function_do


def update_var_fun(graph, functions, variables, observational_samples, xi_dict_var):

    def compute_var(num_interventions, x, xi_dict_var, compute_do):
        var_do = np.zeros((num_interventions, 1))
        for i in range(num_interventions):
            xi_str = str(x[i])
            if xi_str in xi_dict_var:
                var_do[i] = xi_dict_var[xi_str]
            else:
                _, var_do[i] = compute_do(observational_samples, functions, value = x[i])
                xi_dict_var[xi_str] = var_do[i]

        return var_do

    do_functions = graph.get_all_do()
    function_name = get_do_function_name(variables)

    def var_function_do(x):
        num_interventions = x.shape[0]    
        var_do = compute_var(num_interventions, x, xi_dict_var[variables], do_functions[function_name])
        return np.float64(var_do)

    return var_function_do
