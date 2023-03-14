import logging
from omegaconf import OmegaConf
from hydra.utils import instantiate
import hydra


@hydra.main(config_path="config", config_name="run_exploration_set_algo")
def run_exploration_set_algorithm(config):
    """
    Run the exploration set algorithm
    :param config: the hydra configuration
    """

    # Log the hydra configuration
    logging.info("Graph config:\n{}".format(OmegaConf.to_yaml(config)))

    # Instantiate the graph described in the configuration
    graph = instantiate(config.graph)

    # Display the loaded graph
    graph.save_drawing(config.save_file, show=True)


if __name__ == '__main__':
    """
    Script running the exploration set algorithm.
    """
    run_exploration_set_algorithm()
