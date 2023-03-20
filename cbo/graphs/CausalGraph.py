import copy
import uuid

import pandas as pd
from cbo.graphs import logger
from cbo.graphs.Node import Node
from cbo.graphs.StructuralEquations import StringEquation
from cbo.utils_functions.utils import is_valid_path, save_figure, remove_node_from_family, safe_add
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from cbo.algorithms.MIS import MIS


class CausalGraph:
    """
    A class implementing a generic causal graph.
    """

    def __init__(self, name, nodes, interventions_path, obs_path, true_obs_path=None, n_initial_samples=100,
                 exploration_set_fn=MIS()):
        """
        Construct the graph requested in the hydra configuration
        :param name: the graph's name
        :param nodes: the graph's nodes
        :param interventions_path: the path to the numpy file containing the interventions
        :param obs_path: the path to the pickle file containing the measurements
        :param true_obs_path: the path to the pickle file containing the true measurements
        :param n_initial_samples: the initial number of observations the causal graph has access to
        """

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

        # Create the networkx graph corresponding to the causal graph
        self._graph = None
        self._preprocess_nodes(nodes, name)

        # Create the caches for the c-components and confounded variables
        self.c_components_cache = self._get_c_components()
        # TODO[lisa] |=> for now these caches below are unused
        # TODO[lisa] the self._graph.graph properties should be kept when self.__getitem__ and self.do are called

        # Retrieve the reward variables and the exploration set for the graph
        self.exploration_set = exploration_set_fn.run(self.reward_variables)
        # TODO[lisa] remove graph from ini of MIS/POMIS and update the algo accordingly

    @property
    def name(self):
        """
        Getter
        :return: the graph's name
        """
        return self._graph.graph["name"]

    @property
    def nodes(self):
        """
        Getter
        :return: the graph's nodes
        """
        return self._graph.nodes

    @property
    def edges(self):
        """
        Getter
        :return: the graph's edges
        """
        return self._graph.edges

    @property
    def manipulative_variables(self):
        """
        Getter
        :return: the graph's manipulative variables
        """
        return self._graph.graph["manipulative_variables"]

    @property
    def reward_variables(self):
        """
        Getter
        :return: the reward variables of the graph
        """
        return self._graph.graph["reward_variables"]

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

    def parents(self, nodes):
        """
        Getter
        :params nodes: the names of the nodes whose parents should be returned
        :return: the names of the nodes' parents
        """

        # Turns single node into a list of size one
        if not isinstance(nodes, list):
            nodes = [nodes]

        # Collect all the node's parents
        pa = []
        for node in nodes:
            pa.extend(self._graph.predecessors(self._graph.graph[node]))
        return set(pa)

    def children(self, nodes):
        """
        Getter
        :params nodes: the names of the nodes whose children should be returned
        :return: the names of the nodes' children
        """

        # Turns single node into a list of size one
        if not isinstance(nodes, list):
            nodes = [nodes]

        # Collect all the node's children
        ch = []
        for node in nodes:
            ch.extend(self._graph.successors(self._graph.graph[node]))
        return set(ch)

    def ancestors(self, nodes):
        """
        Getter
        :params nodes: the names of the nodes whose ancestors should be returned
        :return: the names of nodes' ancestors
        """

        # Turns single node into a list of size one
        if not isinstance(nodes, list):
            nodes = [nodes]

        # Retrieve all the nodes ancestors
        ancestors = []
        for node in nodes:
            ancestors.extend(nx.ancestors(self._graph, node))

        return set(ancestors)

    def descendants(self, nodes):
        """
        Getter
        :params nodes: the names of the nodes whose descendants should be returned
        :return: the names of the nodes' descendants
        """

        # Turns single node into a list of size one
        if not isinstance(nodes, list):
            nodes = [nodes]

        # Retrieve all the nodes descendants
        descendants = []
        for node in nodes:
            descendants.extend(nx.descendants(self._graph, node))

        return set(descendants)

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
        return {node.name: c_component for c_component in c_components for node in c_component}

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
            nodes += set(self._graph.graph.confounded_variables[next_node.name]) - c_component
        return c_component

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

    def _preprocess_nodes(self, nodes, name):
        """
        Add the parents and children nodes to all the graph's nodes,
        fit all the nodes' equation, sort the nodes in topological order, and initialise the networkx graph
        :param nodes: the list of nodes to use
        :param: the name of the graph
        """

        nodes_dict = {node.name: node for node in nodes}
        manipulative_variables = [node.name for node in nodes_dict if node.min_intervention is not None]
        edges = [(node.name, child_name) for node in self.nodes for child_name in node.children_name]
        reward_variables = [node_name for node_name, node in nodes_dict.items() if node.is_reward()]
        # TODO[lisa]: take into account the unseen variables specified in the config file here.
        confounded_variables = {n: set() for n in nodes_dict.keys()}

        self._graph = nx.DiGraph(edges, **nodes_dict, manipulative_variables=manipulative_variables, name=name,
                                 confounded_variables=confounded_variables,
                                 reward_variables=reward_variables)
        self._graph.add_nodes_from(nodes_dict.keys())
        self._graph.graph["topological_order"] = list(self.topological_sort())

        # Fit all the nodes' equations
        for node_name in self._graph.graph["topological_order"]:
            # Fit the node's equation based on available measurements
            self._fit_equation(node_name)

    @staticmethod
    def only(nodes, target_set):
        """
        Returns all the nodes that are in the set passed as parameter
        :param nodes: the nodes that needs to be checked
        :param target_set: the target set
        :return: all the nodes that are in the set passed as parameter
        """
        if not target_set:
            return []
        return [node for node in nodes if node in target_set]

    def _bidirected_edges_to_unseen_variables(self):
        """ Convert any bidirected edge to unidirected edge from an unobserved confounder variable.
        For example A <-> B will be transformed to A <- U1 -> B where U1 is the unobserved variable.
        """
        # TODO[lisa]: This does not work when the unuobserved variable is linked to more than
        # TODO[lisa]: two latents.
        # Try to assign a readable name to unobserved variables, if not possible use a unique one.
        base_name = "U" if not len([x for x in self._graph.nodes if "U" in x]) else str(uuid.uuid4())
        equation = StringEquation("lambda epsilon: epsilon")
        i = 0
        while not nx.is_directed_acyclic_graph(self._graph):
            cycle = nx.find_cycle(self._graph)

            if len(cycle) > 2:
                raise ValueError("A directed acyclic graph cannot contain cycles.")

            unobserved_node_name = "{}{}".format(base_name, i)
            logger.warning("Removing bidirected edge between {} and {} and replacing it by unseen variable {} ~ N(0,1).\
            Please check your configuration if this was not intended.".format(cycle[0][0], cycle[0][1],
                                                                              unobserved_node_name))
            n1, n2 = self._graph.graph[cycle[0][0]], self._graph.graph[cycle[0][1]]
            self._graph.graph.confounded_variables[n1.name].add(n2)
            self._graph.graph.confounded_variables[n2.name].add(n1)
            self._graph.graph.confounded_variables[unobserved_node_name] = set()
            unobserved_node = Node(name=unobserved_node_name, is_unobserved=True, equation=equation,
                                   children_name=[n1.name, n2.name])

            # Remove parentage between the nodes with bi-directed edge and replace by an unobserved node
            remove_node_from_family(n1, n2, unobserved_node)
            remove_node_from_family(n2, n1, unobserved_node)

            # Do the same in the networkx graph
            self._graph.remove_edges_from(cycle)
            self._graph.add_edges_from([(unobserved_node_name, n1.name), (unobserved_node_name, n2.name)])
            self._graph.graph[unobserved_node_name] = unobserved_node
            i += 1

    def topological_sort(self, backward=False):
        """
        Sort the nodes of in topological order, i.e., the parents comes before their children
        :param backward: reverse the topological order such that the children comes before their parents
        :return: a list of nodes' name sorted in topological order
        """
        # Replace bi-directed edges by unseen variables if needed
        if not nx.is_directed_acyclic_graph(self._graph):
            self._bidirected_edges_to_unseen_variables()

        # Get the topological ordering
        topological_ordering = nx.topological_sort(self._graph)

        # Reverse it if requested
        if backward:
            topological_ordering = reversed(topological_ordering)

        return topological_ordering

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

    def _fit_equation(self, node):
        """
        Fit the equation of a node based on the true observations is available
        :param node: The name of the node to update
        """
        logger.debug("Fitting structural equation of node {}".format(node))
        if self._true_observations is None:
            return
        node_values = np.ones((1, 1)) * np.array(self._true_observations[node])
        parents = self.parents(node)
        parents_values = np.ones((1, 1)) * np.hstack([self._true_observations[p] for p in parents]) if parents else None
        self._graph.graph[node].fit_equation(node_measurement=node_values, parents_measurements=parents_values)

    def fit_all_gaussian_processes(self, *args, **kwargs):
        # TODO: implement this if still needed
        pass

    def sample(self):
        """
        Sample values from each node of the graph
        """
        for node in self._graph.graph["topological_order"]:
            parents = [self._graph.graph[p] for p in self.parents(node)]
            self._graph.graph[node].structural_equation(parents)
