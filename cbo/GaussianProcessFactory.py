from GPy.core import Mapping
from emukit.model_wrappers import GPyModelWrapper
from cbo.utils_functions.causal_kernels import CausalRBF
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from enum import IntEnum


class GaussianProcessType(IntEnum):
    """
    An enumeration of the Gaussian process
    """
    GRAPH_GP = 0
    CAUSAL_GP = 1
    NON_CAUSAL_GP = 2


class GaussianProcessFactory:
    """
    A class allowing the creation of Gaussian processes
    """

    @staticmethod
    def create(gp_type, x, y, parameters=None, emukit_wrapper=False):
        """
        Create the Gaussian process corresponding to the type passed as parameters
        :param gp_type: the type of Gaussian process to create
        :param x: the input data
        :param y: the output data
        :param parameters: the parameters of the Gaussian process to create
        :param emukit_wrapper: whether to wrap the Gaussian process to make it compatible with the emukit package
        :return: the created Gaussian process
        """

        # A dictionary mapping Gaussian process type to the function creating the associated Gaussian process.
        gp_functions = {
            GaussianProcessType.GRAPH_GP: GaussianProcessFactory.create_graph_gp,
            GaussianProcessType.CAUSAL_GP: GaussianProcessFactory.create_causal_gp,
            GaussianProcessType.NON_CAUSAL_GP: GaussianProcessFactory.create_non_causal_gp
        }

        # Create the requested Gaussian process.
        gp = gp_functions[gp_type](x, y, parameters)

        # Wrap the GPy model to make it compatible with the emukit package, if requested by the user.
        return GPyModelWrapper(gp) if emukit_wrapper is True else gp

    @staticmethod
    def create_graph_gp(x, y, parameters):
        # GRAPH_GAUSSIAN_PROCESS
        kernel = RBF(x.shape[1], ARD=parameters[3], lengthscale=parameters[0], variance=parameters[1])
        gp = GPRegression(x, y, kernel, noise_var=parameters[2])
        gp.likelihood.variance.fix(1e-2)
        return gp

    @staticmethod
    def create_non_causal_gp(x, y, _):
        # NON_CAUSAL_GAUSSIAN_PROCESS
        kernel = RBF(x.shape[1], lengthscale=1., variance=1.)
        return GPRegression(x, y, kernel, noise_var=1e-10)

    @staticmethod
    def create_causal_gp(x, y, parameters):
        # CAUSAL_GAUSSIAN_PROCESS
        mean_function, var_function = parameters

        mf = Mapping(x.shape[1], 1)
        mf.f = mean_function
        mf.update_gradients = lambda a, b: None
        causal_kernel = CausalRBF(
            x.shape[1], variance_adjustment=var_function, lengthscale=1., variance=1., ARD=False
        )
        return GPRegression(x, y, causal_kernel, noise_var=1e-10, mean_function=mf)
