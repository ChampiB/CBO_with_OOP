from operator import itemgetter

from src.graphs import GraphInterface


class Graph(GraphInterface):

    def __init__(self, nodes, observation_path, intervention_path, true_observation_path=None, seed=0):
        # This created a mapping between node names and node indexes
        # to avoid iterating over all nodes when we need a node with a specific name
        self._nodes, self._nodes_map = self._preprocess_nodes(nodes)
        self._manipulative_variables = [n for n in nodes if n.min_intervention is not None]
        self._seed = seed

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
        pass

    def get_exploration_set(self):
        pass

    def sample(self):
        """Sample values from each nodes of the graph

        :return: None
        """
        for node in self.nodes:
            node.structural_equation()

    def _preprocess_nodes(self, nodes):
        """Add parents and children nodes to all the nodes then reorder them.

        :param nodes: the list of nodes
        :return: the ordered nodes and a dict mapping the node names and indexes in the ordered list
        """
        ordered_nodes = []
        ordered_nodes_names = []

        for node in nodes:
            node.parents = itemgetter(node.parents_name)
            node.children = itemgetter(node.children_name)

        while len(nodes) > 0:
            self._sort_nodes(nodes, ordered_nodes, ordered_nodes_names)
        node_map = {n: i for n, i in enumerate(ordered_nodes_names)}

        return ordered_nodes, node_map

    @staticmethod
    def _sort_nodes(nodes, ordered_nodes, ordered_nodes_names):
        """ Order the list of nodes so that the parent of each node is before it in
        the node list. This is required for the sampling process.

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
