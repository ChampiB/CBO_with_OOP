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

    def update_do_function(self, gaussian_processes, intervention, index, values):
        """
        Compute the mean and variance functions of an intervention
        :param index: if index == 0, then compute mean function, else compute variance function
        :param intervention: the intervention whose functions need to be computed
        :param gaussian_processes: the previous functions
        :param values: the inputs
        :return: the new mean and variance functions
        """

        # Get the do function and the mean or variance functions  # TODO right?
        functions = self.cbo.x_mean[intervention] if index == 0 else self.cbo.x_var[intervention]
        name = self.cbo.graph.get_gp_name(intervention)
        gp = gaussian_processes[name]

        # Create an array of variances
        do_function = np.zeros((values.shape[0], 1))
        for i, value in enumerate(values):

            # Compute the variance of the i-th intervention
            value_name = str(value)
            if value_name in functions.keys():
                do_function[i] = functions[value_name]
            else:
                input_var = next(filter(lambda dep: dep[0] == intervention[0], self.cbo.graph.fit_dependencies))
                do_function[i] = self.compute_do(self.cbo.measurements, gp, value, input_var, intervention)[index]
                do_function[i] = np.mean(do_function[i])

            # Store the mean in the directory of variances
            if value_name not in functions.keys():
                functions[value_name] = do_function[i]

        return np.float64(do_function)

    def compute_do(self, measurements, gp, value, input_vars, intervention_vars):
        """
        input_vars: the all the input variables
        intervention_vars: the variables on which we intervene
        """

        intervened_inputs = np.hstack([
            self.get_intervened_inputs(measurements, input_var, intervention_vars, value) for input_var in input_vars
        ])
        return gp.predict(intervened_inputs)

    @staticmethod
    def get_intervened_inputs(measurements, input_var, intervention_vars, value):
        """
        Getter
        :returns: the intervened inputs
        """
        intervened_inputs = np.asarray(measurements[input_var])[:, np.newaxis]
        if input_var in intervention_vars:
            index = intervention_vars.index(input_var)
            intervened_inputs = np.ones_like(intervened_inputs) * value[index]
        return intervened_inputs
