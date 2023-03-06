import logging
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
logger = logging.getLogger("test_graph")


@hydra.main(config_path="config", config_name="test_graph")
def test_graph(cfg):
    logger.info("Graph config:\n{}".format(OmegaConf.to_yaml(cfg)))
    graph = instantiate(cfg.graph)
    graph.show(cfg.save_file, show=True)


if __name__ == "__main__":
    test_graph()
