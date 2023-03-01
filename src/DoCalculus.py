import numpy as np


class DoCalculus:

    def __int__(self):
        pass

    @staticmethod
    def get_function_name(interventions):
        return 'compute_do_' + "".join(interventions)

    @staticmethod
    def update_mean_fun(graph, functions, variables, observational_samples, xi_dict_mean):
        do_functions = graph.get_all_do()
        function_name = DoCalculus.get_function_name(variables)

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

    @staticmethod
    def update_var_fun(graph, functions, variables, observational_samples, xi_dict_var):

        function_name = DoCalculus.get_function_name(variables)
        do_function = graph.get_all_do()[function_name]

        def var_function_do(x):
            num_interventions = x.shape[0]
            xi_var = xi_dict_var[variables]

            var_do = np.zeros((num_interventions, 1))
            for i in range(num_interventions):
                xi = str(x[i])
                if xi in xi_var:
                    var_do[i] = xi_var[xi]
                else:
                    var_do[i] = do_function(observational_samples, functions, value=x[i])[1]
                    xi_var[xi] = var_do[i]

            return np.float64(var_do)

        return var_function_do
