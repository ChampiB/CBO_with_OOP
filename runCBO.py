from src.CBO import *
from src.ArgumentParser import ArgumentParser as ArgumentParser
from src.DataLoader import DataLoader as DataLoader


if __name__ == '__main__':
    """
    Script running the causal Bayesian optimisation algorithm.
    """
    # Parse and display the argument of the script.
    args = ArgumentParser().parse(verbose=True)

    # Load the measurements and associated graph.
    data = DataLoader(args.experiment, args.initial_num_obs_samples)

    # Run the causal Bayesian optimisation.
    cbo_agent = CBO(args, data, verbose=True)
    cbo_agent.run()
