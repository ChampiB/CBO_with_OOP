import abc


class GraphStructure:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def define_sem(self):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def fit_all_gaussian_processes(self):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def fit_all_gaussian_processes(self, observational_samples):
        raise NotImplementedError("Subclass should implement this.")

    @abc.abstractmethod
    def get_do_function(self, function_name):
        raise NotImplementedError("Subclass should implement this.")

    @staticmethod
    def get_function_name(interventions):
        """
        Getter
        :param interventions: the interventions for which the do-function's name is returned
        :return: the do-function's name
        """
        return 'compute_do_' + "".join(interventions)

    @staticmethod
    def get_gp_name(interventions):
        """
        Getter
        :param interventions: the interventions for which the Gaussian process' name is returned
        :return: the Gaussian process' name
        """
        return 'gp_' + "_".join(interventions)
