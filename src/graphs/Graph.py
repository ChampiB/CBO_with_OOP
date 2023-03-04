from operator import itemgetter
import pandas as pd
from src.utils_functions import is_valid_path
import numpy as np


class Graph:

    def __init__(self, nodes, observation_path, intervention_path, initial_num_obs_samples=100,
                 true_observation_path=None, seed=0):
        manipulative_variables = [n for n in nodes if n.min_intervention is not None]
        self._manipulative_variables = manipulative_variables
        self._nodes, self._nodes_map = self._preprocess_nodes(nodes)
        self._seed = seed
        # This replace DataLoader
        self._observations = pd.read_pickle(observation_path)[:initial_num_obs_samples]
        self._true_observations = pd.read_pickle(true_observation_path) if is_valid_path(true_observation_path) else None
        self._interventions = np.load(intervention_path, allow_pickle=True)

    @property
    def nodes(self):
        return self._nodes

    @property
    def nodes_map(self):
        return self._nodes_map

    @property
    def manipulative_variables(self):
        return self._manipulative_variables

    def fit_all_gaussian_processes(self):
        # TODO: implement this if still needed
        pass

    def fit_all_gaussian_processes(self, observational_samples):
        # TODO: implement this if still needed
        pass

    def get_do_function(self, function_name):
        # TODO: implement this if still needed
        pass

    def get_cost_structure(self, type_cost):
        # TODO: implement this if still needed
        pass

    def get_exploration_set(self):
        # TODO: implement this
        pass

    def sample(self):
        """Sample values from each nodes of the graph

        :return: None
        """
        for node in self.nodes:
            node.structural_equation()

    def _preprocess_nodes(self, nodes):
        """Add parents and children nodes to all the nodes then reorder them so a node is added to the list only
        when all its parents are already in the list.

        :param nodes: the list of nodes
        :return: the ordered nodes and a dict mapping the node names and indexes in the ordered list
        """
        ordered_nodes = []
        ordered_nodes_names = []

        for node in nodes:
            node.parents = itemgetter(node.parents_name)
            node.children = itemgetter(node.children_name)
            self._fit_equation(node)

        while nodes:
            self._sort_nodes(nodes, ordered_nodes, ordered_nodes_names)
        node_map = {n: i for n, i in enumerate(ordered_nodes_names)}

        return ordered_nodes, node_map

    def _fit_equation(self, node):
        """ Fit the equation of a node based on the true observations is available

        :param node: The node to update
        :return: None
        """
        if self._true_observations is None:
            return
        node_values = self._true_observations[node.name]
        parents_values = np.hstack([self._true_observations[p.name] for p in node.parents]) if node.parents else None
        node.fit_equation(node_measurement=node_values, parents_measurements=parents_values)

    @staticmethod
    def _sort_nodes(nodes, ordered_nodes, ordered_nodes_names):
        """ Order the list of nodes so that the parent of each node is already in the list when we append a node.
        This allow the sampling process to be done in one go, by iterating over the ordered list.

        :param nodes: a list of nodes
        :param ordered_nodes: the ordered list of nodes
        :param ordered_nodes_names: the names of the nodes in the ordered list
        :return: None
        """
        for node in nodes:
            if all([p in ordered_nodes_names for p in node.parents_name]):
                nodes.remove(node)
                ordered_nodes.append(node)
                ordered_nodes_names.append(node.name)
