import abc
import sklearn.mixture as sm
from scipy.stats import gamma


class StructuralEquationInterface:
    __metaclass__ = abc.ABCMeta

    def __init__(self, equation):
        self._equation = equation

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError("Missing fit method.")

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        raise NotImplementedError("Missing predict method.")


class Gamma(StructuralEquationInterface):
    def __init__(self):
        super(Gamma, self).__init__(gamma)

    def fit(self, *args, **kwargs):
        a, loc, scale = self._equation.fit(args[0][0])
        self._equation = self._equation(a=a, loc=loc, scale=scale)

    def predict(self, *args, **kwargs):
        return self._equation.rvs(1)[0]


class GaussianMixture(StructuralEquationInterface):
    def __init__(self, n_components=3):
        super(GaussianMixture, self).__init__(sm.GaussianMixture(n_components=n_components))

    def fit(self, *args, **kwargs):
        self._equation.fit(args[0][0].reshape(-1, 1))

    def predict(self, *args, **kwargs):
        return self._equation.sample(1)[0][0][0]


class StringEquation(StructuralEquationInterface):
    def __init__(self, string_equation):
        super(StringEquation, self).__init__(eval(string_equation))

    def fit(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        return self._equation(**kwargs)


class SklearnEquation(StructuralEquationInterface):
    def __init__(self, sklearn_model):
        super(SklearnEquation, self).__init__(sklearn_model)

    def fit(self, *args, **kwargs):
        self._equation.fit(*args)

    def predict(self, *args, **kwargs):
        return self._equation.predict(*args)
