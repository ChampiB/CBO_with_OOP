import abc
from abc import ABC


class ExplorationSetInterface(ABC):
    """
    An interface that must be implemented by all exploration set algorithm.
    """

    @abc.abstractmethod
    def __call__(self, graph, reward_variables):
        """
        Compute the exploration set w.r.t. the reward variables
        :param graph: the graph on which the algorithm must be run
        :param reward_variables: the reward variables
        :return: the exploration set
        """
        ...

    @staticmethod
    def set_minus(nodes, names):
        """
        A function selecting only the nodes whose name are in the list of names passed as parameters
        :param nodes: all the nodes that should be filtered out
        :param names: the list of names to keep
        :return: the nodes whose name are in the list of names passed as parameters
        """
        return [node for node in nodes if node not in names]
