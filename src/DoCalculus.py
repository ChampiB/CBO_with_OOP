import numpy as np
from functools import partial


class DoCalculus:

    def __init__(self, cbo):
        self.cbo = cbo

    def update_all_do_functions(self, gaussian_processes):
        """
        Compute the new mean and variance functions
        :param gaussian_processes: the previous functions
        :return: the new mean and variance functions
        """
        return [self.update_functions(index, gaussian_processes) for index in [0, 1]]

    @staticmethod
    def get_function_name(interventions):
        """
        Getter
        :param interventions: the interventions for which the do-function's name is returned
        :return: the do-function's name
        """
        return 'compute_do_' + "".join(interventions)

    def update_functions(self, index, gaussian_processes):
        return [
            self.update_function(DoCalculus.function_do, gaussian_processes, self.cbo.interventions[j], index)
            for j in range(self.cbo.es_size)
        ]

    def update_function(self, function, gaussian_processes, interventions, index):
        function_name = DoCalculus.get_function_name(interventions)
        compute_do = self.cbo.graph.get_do_function(function_name)
        return partial(function, self.cbo.x_mean[interventions], compute_do, gaussian_processes, index=index)

    def function_do(self, gp, compute_do, functions, xs, index):
        # Create an array of variances
        var_do = np.zeros((xs.shape[0], 1))
        for i, x in enumerate(xs):

            # Compute the variance of the i-th intervention
            name = str(x)
            var_do[i] = gp[name] if name in gp.keys() else compute_do(self.cbo.measurements, functions, value=x)[index]

            # Store the mean in the directory of variances
            if name not in gp.keys():
                gp[name] = var_do[i]

        return np.float64(var_do)
