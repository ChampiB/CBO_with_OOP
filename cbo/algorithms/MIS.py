from cbo.algorithms.ExplorationSetInterface import ExplorationSetInterface


class MIS(ExplorationSetInterface):
    """
    An implementation of the Minimal Intervention Sets algorithm.
    """

    def __init__(self, **kwarg):
        pass

    def run(self, graph, reward_variables):
        """
        Find all the minimal intervention sets w.r.t. the reward variables
        :param graph: the graph on which the algorithm must be run
        :param reward_variables: the reward variables
        :return: all the minimal intervention sets
        """

        # Collect all the nodes that does not correspond to reward variables
        non_reward_nodes = self.filter_set(graph.nodes, reward_variables)

        # Create the graph induced by the ancestors of the reward variables
        induced_graph = graph[graph.ancestors(reward_variables)]

        # Create a topological ordering of all the non-reward nodes
        topological_ordering = induced_graph.topological_sort(backward=True)
        topological_ordering = graph.only(topological_ordering, non_reward_nodes)

        # Compute all the minimal intervention sets recursively
        return self.sub_miss(induced_graph, reward_variables, frozenset(), topological_ordering)

    def sub_miss(self, graph, reward_variables, mis, topological_ordering):
        """
        Builds a set of minimal intervention sets by adding a variable to a previously obtained minimal intervention set
        :param graph: the graph whose minimal intervention sets should be returned
        :param reward_variables: the reward variables
        :param mis: a previously found minimum intervention set
        :param topological_ordering: the nodes to add to the minimal intervention sets in topological order
        """

        # Create the new set of minimal intervention sets
        new_miss = {mis}
        for i, node in enumerate(topological_ordering):

            # Create the graph obtained when intervening on the next node of the topological ordering
            do_graph = graph.do({node})

            # Create the graph induced by the ancestors of the reward variables
            do_graph = do_graph[do_graph.ancestors(reward_variables)]

            # Create the new topological ordering
            new_topological_ordering = self._graph.only(topological_ordering[i + 1:], do_graph.nodes)

            # Compute all the sub-minimal intervention sets by adding a variable to a previously obtained
            # minimal intervention set
            new_miss |= self.sub_miss(do_graph, reward_variables, mis | {node}, new_topological_ordering)

        return new_miss
