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

