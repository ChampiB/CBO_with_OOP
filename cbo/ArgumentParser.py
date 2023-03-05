from numpy import random
from argparse import ArgumentParser as ArgParser


class ArgumentParser:
    """
    A class used to parse the argument of the script called: 'runCBO.py'.
    """

    def __init__(self):
        """
        Create the parser
        """
        parser = ArgParser(description='This script running the Causal Bayesian Optimisation (CBO) algorithm.')
        help_msg = "An integer representing the initial number of observational samples."
        parser.add_argument('--initial_num_obs_samples', default=100, type=int, help=help_msg)
        help_msg = "An integer representing the size of the initial interventional dataset."
        parser.add_argument('--num_interventions', default=10, type=int, help=help_msg)
        help_msg = "An integer representing the type of cost per node, i.e., "
        help_msg += "fix_equal = 1, fix_different = 2, fix_different_variable = 3, fix_equal_variable = 4."
        parser.add_argument('--type_cost', default=1, type=int, help=help_msg)
        help_msg = "An integer representing the number of additional observations collected for every decision."
        parser.add_argument('--num_additional_observations', default=20, type=int, help=help_msg)
        help_msg = "An integer representing the number of BO trials"
        parser.add_argument('--num_trials', default=40, type=int, help=help_msg)
        help_msg = "An integer index of interventional dataset used."
        parser.add_argument('--name_index', default=0, type=int, help=help_msg)
        help_msg = "An integer representing the seed to use for the experiment"
        parser.add_argument('--seed', default=9, type=int, help=help_msg)
        parser.add_argument('--exploration_set', default='MIS', type=str, help='exploration set')
        parser.add_argument('--causal_prior', default=False, type=bool, help='Do not specify when want to set to False')
        parser.add_argument('--experiment', default='complete_graph', type=str, help='experiment')
        parser.add_argument('--task', default='min', type=str, help='experiment')
        self.parser = parser

    def parse(self, verbose=False):
        """
        Parse the arguments
        :param verbose: True if the argument loaded needs to be displayed, False otherwise
        :return: the script arguments
        """

        # Parse the arguments
        args = self.parser.parse_args()

        # Set the seed for reproducibility
        random.seed(args.seed)

        # Display loaded arguments, if requested
        if verbose is True:
            print('================================== Parsed arguments ==================================')
            print('exploration_set', args.exploration_set)
            print('initial_num_obs_samples', args.initial_num_obs_samples)
            print('num_interventions', args.num_interventions)
            print('type_cost', args.type_cost)
            print('num_trials', args.num_trials)
            print('causal_prior', args.causal_prior)
            print('experiment', args.experiment)
            print('task', args.task)
            print('======================================================================================')
            print()
        return args
