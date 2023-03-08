import copy
import pandas as pd
from omegaconf import DictConfig
from cbo.graphs import logger
from cbo.utils_functions.utils import is_valid_path, save_figure
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class CausalGraph:
    """
    A class implementing a generic causal graph.
    """

    def __init__(self, name, nodes, interventions_path, obs_path, true_obs_path=None, n_initial_samples=100):
        """
        Construct the graph requested in the hydra configuration
        :param name: the graph's name
        :param nodes: the graph's nodes
        :param interventions_path: the path to the numpy file containing the interventions
        :param obs_path: the path to the pickle file containing the measurements
        :param true_obs_path: the path to the pickle file containing the true measurements
        :param n_initial_samples: the initial number of observations the causal graph has access to
        """

        # TODO add exploration set algorithm to hydra configuration, i.e., POMIS vs MIS

        # The networkx graph corresponding to the causal graph
        self._nodes = nodes
        self._edges = self._get_edges()
        self._graph = nx.DiGraph(self._edges)
        self._graph.add_nodes_from(self._nodes)

        # Store the paths of the files containing the measurements and interventions
        self.interventions_path = interventions_path
        self.obs_path = obs_path
        self.true_obs_path = true_obs_path

        # Store the initial number of samples the causal graph has access to
        self.n_initial_samples = n_initial_samples

        # Load the measurements and interventions
        self._interventions = np.load(interventions_path, allow_pickle=True)
        self._observations = pd.read_pickle(obs_path)[:n_initial_samples]
        self._true_observations = pd.read_pickle(true_obs_path) if is_valid_path(true_obs_path) else None

        # The graph's name
        self._name = name

        # Pre-process the nodes and retrieve the nodes map as well as the manipulative variables
        self._preprocess_nodes()
        self._nodes_map = self._get_node_map()
        self._manipulative_variables = [node for node in nodes if node.min_intervention is not None]

        # Create an empty list of bi-directed edges
        self.bi_directed_edges = []
        # TODO add support for bi_directed_edges and unobserved confounders
        # TODO bi-directed edges are 3-tuples of the form (x:Node, y:Node, u:confounder_name -> a string)
        # TODO update comment above self.bi_directed_edges

        # Create the caches for the ancestors, descendants, c-components and confounded variables
        self.ancestors_cache = {}
        self.descendants_cache = {}
        self.confounded_vars_cache = self._get_confounded_variables()
        self.c_components_cache = self._get_c_components()

        # TODO cache the ancestors and descendants variables for faster access (using a cache decorator?)
        # TODO |=> for now these caches below are unused
        # TODO all 4 caches should also be transferred when self.__getitem__ and self.do are called

    def __getitem__(self, nodes):
        """
        Create the graph induced by the nodes passed as parameters
        :param nodes: the nodes for which the induced graph must be returned
        :return: the induced graph
        """
        return CausalGraph(
            self.name,
            self._format_induced_nodes(nodes),
            interventions_path=self.interventions_path,
            obs_path=self.obs_path,
            true_obs_path=self.true_obs_path,
            n_initial_samples=self.n_initial_samples
        )

    @staticmethod
    def _format_induced_nodes(nodes):
        """
        Format the nodes passed as parameters to fit the format expected by the causal graph constructor
        :params nodes: the nodes to format
        :return: the formatted nodes
        """

        # Get the name of all nodes passed as parameters
        nodes_name = [node.name for node in nodes]

        # Format all nodes passed as parameters
        formatted_nodes = []
        for node in nodes:
            new_node = copy.deepcopy(node)
            new_node.parents_name = [parent_name for parent_name in node.parents_name if parent_name in nodes_name]
            new_node.children_name = [child_name for child_name in node.children_name if child_name in nodes_name]
            new_node.parents = []
            new_node.children = []
            formatted_nodes.append(new_node)

        return formatted_nodes

    def do(self, nodes):
        """
        Create the graph obtained by intervening on the nodes passed as parameters
        :param nodes: the nodes for which the induced graph must be returned
        :return: the graph obtained by intervening on the nodes passed as parameters
        """
        return CausalGraph(
            self.name,
            self._format_do_nodes(nodes),
            interventions_path=self.interventions_path,
            obs_path=self.obs_path,
            true_obs_path=self.true_obs_path,
            n_initial_samples=self.n_initial_samples
        )

    def _format_do_nodes(self, do_nodes):
        """
        Format all the graph's nodes to fit the format expected by the causal graph constructor
        :param do_nodes: all the nodes being intervened on
        :return: the formatted nodes
        """

        # Get the name of all nodes being intervened on
        do_nodes_name = [node.name for node in do_nodes]

        # Format all the graph's nodes
        formatted_nodes = []
        for node in self.nodes:

            # Make a copy of each node, and get its name
            new_node = copy.deepcopy(node)
            new_node_name = new_node.name

            # Remove all the parents of the node, the node is intervened on
            if new_node_name in do_nodes_name:
                new_node.parents = []

            # Remove all the node's children that are intervened on
            for child in new_node.children:
                if child.name in do_nodes_name:
                    new_node.children.remove(child)

            # Add the new node to the list of formatted nodes
            formatted_nodes.append(new_node)

        return formatted_nodes

    @staticmethod
    def parents(nodes):
        """
        Getter
        :params nodes: the nodes whose parents should be returned
        :return: the nodes' parents
        """

        # Turns single node into a list of size one
        if not isinstance(nodes, list):
            nodes = [nodes]

        # Collect all the node's parents
        pa = []
        for node in nodes:
            pa.extend(node.parents)
        return pa

    @staticmethod
    def children(nodes):
        """
        Getter
        :params nodes: the nodes whose children should be returned
        :return: the nodes' children
        """

        # Turns single node into a list of size one
        if not isinstance(nodes, list):
            nodes = [nodes]

        # Collect all the node's children
        ch = []
        for node in nodes:
            ch.extend(node.children)
        return ch

    @staticmethod
    def ancestors(nodes, ancestors=None):
        """
        Getter
        :params nodes: the nodes whose ancestors should be returned
        :params ancestors: the current list of ancestors
        :return: the nodes' ancestors
        """

        # Turns single node into a list of size one
        if not isinstance(nodes, list):
            nodes = [nodes]

        # Collect all the node's ancestors
        an = []
        for node in nodes:
            for parent in node.parents:

                # Add the parent and its ancestors to the list of ancestors, if not already added
                if parent not in ancestors:
                    an.append(parent)
                    an += CausalGraph.ancestors(parent, ancestors=an)
        return an

    @staticmethod
    def descendants(nodes, descendants=None):
        """
        Getter
        :params nodes: the nodes whose descendants should be returned
        :params descendants: the current list of descendants
        :return: the nodes' descendants
        """

        # Turns single node into a list of size one
        if not isinstance(nodes, list):
            nodes = [nodes]

        # Collect all the node's descendants
        de = []
        for node in nodes:
            for child in node.children:

                # Add the child and its descendants to the list of descendants, if not already added
                if child not in descendants:
                    de.append(child)
                    de += CausalGraph.descendants(child, descendants=de)
        return de

    def _get_c_components(self):
        """
        Getter
        :return: a dictionary whose keys are the nodes name and the values are corresponding c-components
        """

        # Collect the c-components of all the graph's nodes
        c_components = []
        remaining_nodes = set(self.nodes)
        found_nodes = set()
        while remaining_nodes:

            # Remove a random element from the set of all remaining nodes, and find its c-component
            node = remaining_nodes.pop()
            c_component = self._get_c_component(node)
            c_components.append(c_component)

            # Update the sets of found and remaining nodes
            found_nodes |= c_component
            remaining_nodes -= found_nodes

        # Collect the c-component of each node
        return {node.name: c_components[node.name] for node in self.nodes}

    def _get_c_component(self, node):
        """
        Getter
        :param node: the node whose c-component needs to be returned
        :return: the c-component of the node passed as parameters
        """
        c_component = set()
        nodes = [node]
        while nodes:
            next_node = nodes.pop()
            c_component.add(next_node)
            nodes += set(self.confounded_vars_cache[next_node]) - c_component
        return c_component

    def _get_confounded_variables(self):
        """
        Getter
        :return: a dictionary whose keys are the nodes name and the values are corresponding lists of confounded nodes
        """

        # Create the list of confounded variables associated to all bi-directed edges in the graph
        confounded_vars = {}
        for x, y, u in self.bi_directed_edges:

            # Add y as a confounded variable of x, if not done already
            self._safe_add(confounded_vars, x.name, y)

            # Add x as a confounded variable of y, if not done already
            self._safe_add(confounded_vars, y.name, x)

        # Create an empty list for each variable that is not confounded with any other variables
        for node in self.nodes:
            if node.name not in confounded_vars.keys():
                confounded_vars[node.name] = []

        return confounded_vars

    @staticmethod
    def _safe_add(dictionary, key, new_node):
        """
        Add a new node to the list of nodes corresponding to the key passed as parameters
        :param dictionary: the dictionary whose keys are nodes name and values are the corresponding list of nodes
        :param key: the key for which a new node needs to be added
        :param new_node: the new node to add to the list of nodes corresponding to the key
        """
        if key not in dictionary.keys():
            dictionary[key] = [new_node]
        elif new_node not in dictionary[key]:
            dictionary[key].append(new_node)

    def c_component(self, nodes):
        """
        Getter
        :params nodes: the nodes whose c-component should be returned
        :return: the nodes' c-component
        """

        # Turns single node into a list of size one
        if not isinstance(nodes, list):
            nodes = [nodes]

        # Create the union of the nodes' c-components
        return set.union(*[self.c_components_cache[node.name] for node in nodes])

    @property
    def name(self):
        """
        Getter
        :return: the graph's name
        """
        return self._name

    @property
    def nodes(self):
        """
        Getter
        :return: the graph's nodes
        """
        return self._nodes

    @property
    def edges(self):
        """
        Getter
        :return: the graph's edges
        """
        return self._edges

    @property
    def nodes_map(self):
        """
        Getter
        :return: a map from nodes name to the corresponding nodes in the graph
        """
        return self._nodes_map

    @property
    def manipulative_variables(self):
        """
        Getter
        :return: the graph's manipulative variables
        """
        return self._manipulative_variables

    def _get_edges(self):
        """
        Collect all the edges corresponding to the graph's nodes, i.e., self.nodes
        :return: the graph's edges
        """
        return [(node.name, child_name) for node in self.nodes for child_name in node.children_name]

    def _get_node_map(self):
        """
        Create the nodes map
        :return: a map from nodes name to the corresponding nodes in the graph
        """
        return {node.name: node for node in self._nodes}

    def _preprocess_nodes(self):
        """
        Add the parents and children nodes to all the graph's nodes,
        fit all the nodes' equation, and sort the nodes in topological order
        """

        # This is not ideal but hydra cannot handle list of configs for now so this is a workaround
        if isinstance(self._nodes, DictConfig):
            self._nodes = list(self._nodes.values())

        # Add parents and children nodes to all the nodes, and fit the node's equation
        nodes_map = self._get_node_map()
        for node in self._nodes:

            # Get the node parents and children
            node.parents = [nodes_map[parent_name] for parent_name in node.parents_name]
            node.children = [nodes_map[child_name] for child_name in node.children_name]

            # Log the retrieved parents and children
            self.log_adjacent_nodes(node)

            # Fit the node's equation based on available measurements
            self._fit_equation(node)

        # Perform a topological sort on the graph's nodes
        nodes_name = self.topological_sort()
        self._nodes = [nodes_map[node_name] for node_name in nodes_name]

    @staticmethod
    def _all_nodes_in_set(nodes, target_set):  # TODO remove this? as "only" in MIS may be graph.nodes.name
        """
        Returns all the nodes that are in the target set
        :param nodes: the nodes that needs to be checked
        :param target_set: the target set
        :return: all the nodes that are in the target set
        """
        if not target_set:
            return []
        return [node for node in nodes if node in target_set]

    def topological_sort(self, backward=False):
        """
        Sort the nodes of in topological order, i.e., the parents comes before their children
        :param backward: reverse the topological order such that the children comes before their parents
        :return: a list of nodes' name sorted in topological order
        """
        # Get the topological ordering
        topological_ordering = nx.topological_sort(self._graph)

        # Reverse it if requested
        if backward:
            topological_ordering = reversed(topological_ordering)

        # Turn the topological ordering into a list
        return list(topological_ordering)

    @staticmethod
    def log_adjacent_nodes(node):
        """
        Logs the adjacent nodes to node passed as parameters, i.e., the node's parents and children
        :param node: the node whose parents and children should be displayed
        """
        logger.debug("Getting the parents of node {}".format(node.name))
        logger.debug("node.parents = {}".format(node.parents))
        logger.debug("Getting the children of node {}".format(node.name))
        logger.debug("node.children = {}".format(node.children))

    def save_drawing(self, f_name, show=False):
        """
        Save a drawing of the graph
        :param f_name: the file name where the drawing should be saved
        :param show: whether to show the graph before to save it on the file system
        """
        nx.draw_networkx(self._graph, arrows=True)
        if show:
            plt.show()
        save_figure(f_name)

    # TODO implement and refacto functions below

    def _fit_equation(self, node):
        """
        Fit the equation of a node based on the true observations is available
        :param node: The node to update
        """
        logger.debug("Fitting structural equation of node {}".format(node.name))
        if self._true_observations is None:
            return
        node_values = np.ones((1, 1)) * np.array(self._true_observations[node.name])
        parents_values = np.ones((1, 1)) * np.hstack([self._true_observations[p.name] for p in node.parents]) if node.parents else None
        node.fit_equation(node_measurement=node_values, parents_measurements=parents_values)

    def fit_all_gaussian_processes(self, *args, **kwargs):
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
        """
        Sample values from each node of the graph
        """
        for node in self.nodes:
            node.structural_equation()
