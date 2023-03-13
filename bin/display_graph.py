import logging
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate


@hydra.main(config_path="config", config_name="display_graph")
def display_graph(config):
    """
    Load and display the graph described in the configuration
    :param config: the hydra configuration
    """

    # Log the hydra configuration
    logging.info("Graph config:\n{}".format(OmegaConf.to_yaml(config)))

    # Instantiate the graph described in the configuration
    graph = instantiate(config.graph)

    # Display the loaded graph
    graph.save_drawing(config.save_file, show=True)


if __name__ == "__main__":
    """
    Loading and displaying the graph described in the configuration.
    """
    display_graph()
