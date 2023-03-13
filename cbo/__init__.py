from omegaconf import OmegaConf
from hydra.utils import instantiate

OmegaConf.register_new_resolver("set_cost_value", lambda cost, c1: c1 if (1 < cost < 4) else 1)
OmegaConf.register_new_resolver("set_variable_cost", lambda cost: cost >= 3)
OmegaConf.register_new_resolver("get_values", lambda xs: [instantiate(xs[k]) for k in xs.keys()])
