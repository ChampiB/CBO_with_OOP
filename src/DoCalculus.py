import numpy as np
from functools import partial


class DoCalculus:

    def __init__(self, cbo):
        """
        Create the do-calculus engine
        :param cbo: the casual Bayesian optimisation agent
        """
        self.cbo = cbo

    def update_all_do_functions(self, gaussian_processes):
        """
        Compute the new mean and variance functions
        :param gaussian_processes: the previous functions
        :return: the new mean and variance functions
        """
        return [self.update_do_functions(index, gaussian_processes) for index in [0, 1]]

    def update_do_functions(self, index, gaussian_processes):
        """
        Compute the mean and variance functions for each intervention
        :param index: if index == 0, then compute mean function, else compute variance function
        :param gaussian_processes: the previous functions
        :return: the new mean and variance functions
        """
        return [
            partial(self.update_do_function, gaussian_processes, self.cbo.interventions[i], index)
            for i in range(self.cbo.es_size)
        ]

    def update_do_function(self, gaussian_processes, intervention, index, xs):
        """
        Compute the mean and variance functions of an intervention
        :param index: if index == 0, then compute mean function, else compute variance function
        :param intervention: the intervention whose functions need to be computed
        :param gaussian_processes: the previous functions
        :param xs: the inputs
        :return: the new mean and variance functions
        """
        #
        compute_do = self.cbo.graph.get_do_function(intervention)
        functions = self.cbo.x_mean[intervention] if index == 0 else self.cbo.x_var[intervention]

        # Create an array of variances
        do_function = np.zeros((xs.shape[0], 1))
        for i, x in enumerate(xs):

            # Compute the variance of the i-th intervention
            name = str(x)
            if name in functions.keys():
                do_function[i] = functions[name]
            else:
                do_function[i] = compute_do(self.cbo.measurements, gaussian_processes, value=x)[index]

            # Store the mean in the directory of variances
            if name not in functions.keys():
                functions[name] = do_function[i]

        return np.float64(do_function)
