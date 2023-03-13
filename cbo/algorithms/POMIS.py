from cbo.algorithms.ExplorationSetInterface import ExplorationSetInterface


class POMIS(ExplorationSetInterface):
    """
    An implementation of the Possibly-Optimal Minimal Intervention Sets algorithm.
    """

    def __init__(self, graph):
        """
        Constructor of the possibly optimal minimal intervention sets algorithm
        :param graph: the graph on which the algorithm will be run
        """
        self._graph = graph

    def run(self, reward_variables):
        """
        Find all the possibly optimal minimal intervention sets w.r.t. the reward variables
        :param reward_variables: the reward variables
        :return: all the possibly optimal minimal intervention sets
        """

        # Create the graph induced by the ancestors of the reward variables
        graph = self._graph[self._graph.ancestors(reward_variables)]

        # Compute the minimal unobserved confounder's territory and the interventional border
        territory = self.get_minimal_territories(graph, reward_variables)
        interventional_border = self.interventional_border(graph, territory)

        # Create the graph obtained when intervening on the interventional border
        do_graph = graph.do(interventional_border)

        # Create the graph induced by the nodes in the territory and the interventional border
        induced_graph = do_graph[territory | interventional_border]

        # Create a topological ordering of the nodes in the induced graph
        interventions = induced_graph.topological_sort(backward=True)

        # Only keep the nodes which are in the territory minus the reward variables
        interventions = self._graph.only(interventions, territory - {reward_variables})

        # Compute all the possibly optimal minimal intervention sets recursively
        return self.sub_possibly_optimal_sets(induced_graph, reward_variables, interventions) | {interventional_border}

    @staticmethod
    def get_minimal_territories(graph, reward_variables):
        """
        Compute the minimal unobserved confounder's territory
        :param graph: the graph whose minimal unobserved confounder's territory should be returned
        :param reward_variables: the reward variables for which to compute the minimal unobserved confounder's territory
        :return: the minimal unobserved confounder's territory
        """

        # Create the graph induced by the ancestors of the reward variables
        induced_graph = graph[graph.ancestors(reward_variables)]

        # TODO
        Qs = {reward_variables}
        territories = {reward_variables}
        while Qs:
            Ws = induced_graph.c_component(Qs.pop())
            territories |= Ws
            Qs = (Qs | induced_graph.descendants(Ws)) - territories

        return territories

    @staticmethod
    def interventional_border(graph, territory):
        """
        Compute the interventional border corresponding to the territory passed as parameter
        :param graph: the graph in which the interventional border should be computed
        :param territory: the minimal unobserved confounder's territory
        :return: the interventional border
        """
        return graph.parents(territory) - territory

    def sub_possibly_optimal_sets(self, graph, reward_variables, interventions, obs=None):
        """
        Compute all the possibly optimal minimal intervention sets for a graph with respect to reward variables
        :param graph: the graph whose possibly optimal minimal intervention sets must be returned
        :param reward_variables: the reward variables
        :param interventions: the topological ordering of the graph's nodes
        :param obs: TODO
        :return: all the possibly optimal minimal intervention sets
        """

        if obs is None:
            obs = set()

        possibly_optimal_sets = []
        for i, intervention in enumerate(interventions):
            do_graph = graph.do({intervention})

            # Compute the minimal unobserved confounder's territory and the interventional border
            territory = self.get_minimal_territories(do_graph, reward_variables)
            interventional_border = self.interventional_border(do_graph, territory)

            new_obs = obs | set(interventions[:i])
            if not (interventional_border & new_obs):
                possibly_optimal_sets.append(interventional_border)
                new_interventions = self._graph.only(interventions[i + 1:], territory)

                if new_interventions:
                    new_graph = graph.do(interventional_border)[territory | interventional_border]
                    possibly_optimal_sets.extend(
                        self.sub_possibly_optimal_sets(new_graph, reward_variables, new_interventions, new_obs)
                    )

        return possibly_optimal_sets