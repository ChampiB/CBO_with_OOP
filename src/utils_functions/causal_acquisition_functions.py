import numpy as np
import scipy.stats
import scipy
from emukit.core.interfaces import IDifferentiable
from emukit.core.acquisition import Acquisition


class CausalExpectedImprovement(Acquisition):

    def __init__(self, current_global_min, task, model, jitter=0.0):
        """
        This acquisition computes for a given input the improvement over the current best observed value in
        expectation. For more information see:

        Efficient Global Optimization of Expensive Black-Box Functions
        Jones, Donald R. and Schonlau, Matthias and Welch, William J.
        Journal of Global Optimization

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """
        self.model = model
        self.jitter = jitter
        self.current_global_min = current_global_min
        self.task = task

    def evaluate(self, x):
        """
        Computes the Expected Improvement
        :param x: points where the acquisition is evaluated
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)
        mean += self.jitter

        u, pdf, cdf = get_standard_normal_pdf_cdf(self.current_global_min, mean, standard_deviation)
        if self.task == 'min':
            improvement = standard_deviation * (u * cdf + pdf)
        else:
            improvement = - (standard_deviation * (u * cdf + pdf))

        return improvement

    def evaluate_with_gradients(self, x):
        """
        Computes the Expected Improvement and its derivative
        :param x: locations where the evaluation with gradients is done
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)

        mean += self.jitter
        u, pdf, cdf = get_standard_normal_pdf_cdf(self.current_global_min, mean, standard_deviation)
        
        if self.task == 'min':
            improvement = standard_deviation * (u * cdf + pdf)
            dimprovement_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx
        else:
            improvement = - (standard_deviation * (u * cdf + pdf))
            dimprovement_dx = -(dstandard_deviation_dx * pdf - cdf * dmean_dx)

        return improvement, dimprovement_dx

    @property
    def has_gradients(self):
        """
        Returns that this acquisition has gradients
        """
        return isinstance(self.model, IDifferentiable)


def get_standard_normal_pdf_cdf(x, mean, standard_deviation):
    """
    Returns pdf and cdf of standard normal evaluated at (x - mean)/sigma
    :param x: Non-standardized input
    :param mean: Mean to normalize x with
    :param standard_deviation: Standard deviation to normalize x with
    :return: (normalized version of x, pdf of standard normal, cdf of standard normal)
    """
    u = (x - mean) / standard_deviation
    pdf = scipy.stats.norm.pdf(u)
    cdf = scipy.stats.norm.cdf(u)
    return u, pdf, cdf
