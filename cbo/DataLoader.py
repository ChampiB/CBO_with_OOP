import numpy as np
import os.path
from cbo.graphs import *
import pandas as pd


class DataLoader:
    """
    A class used to load the measurement and causal graph of an experiment.
    """

    def __init__(self, experiment, initial_num_obs_samples):
        """
        Create the loader
        :param experiment: the name of the experiment for which the data is loaded
        :param initial_num_obs_samples: the number of initial observation the CBO agent has to learn from
        """

        # Load the measurements
        self.all_measurements = pd.read_pickle(f'./data/{experiment}/observations.pkl')
        self.measurements = self.all_measurements[:initial_num_obs_samples]
        observations_file = f'./data/{experiment}/true_observations.pkl'
        true_measurements = pd.read_pickle(observations_file) if os.path.exists(observations_file) else None

        # Get true interventional data
        self.interventions = np.load(f'./data/{experiment}/interventional_data.npy', allow_pickle=True)

        # The list a supported graphs
        graph_classes = {
            'toy_graph': ToyGraph,
            'complete_graph': CompleteGraph,
            'coral_graph': CoralGraph,
            'simplified_coral_graph': SimplifiedCoralGraph
        }

        # Create the arguments list a supported graphs
        arguments = [self.measurements]
        if true_measurements is not None:
            arguments.append(true_measurements)

        # Load the causal graph
        self.graph = graph_classes[experiment](*arguments)
