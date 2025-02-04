import copy
import numpy as np
import pandas as pd
from numpy.random import randn
from emukit.core import ParameterSpace, ContinuousParameter


def sample_from_model(model, epsilon=None):
    """ Produces a single sample from a structural equation model (SEM).
    :type model: dict
    :param model: the SEM used for sampling
    :type epsilon: list or None, optional
    :param epsilon: a list of epsilon values used as parameter of some SEMs
    :rtype: dict
    :return: A dictionary of the samples obtained from the SEM in the format {node_name: sample_value}
    """
    epsilon = randn(len(model)) if epsilon is None else epsilon
    # TODO: Find a better way to do this.
    # At the moment **sample is used either as an unused **kwargs or as non-keyword arguments
    # in the SEM functions. Because the SEM is an ordered dict, the behaviour of each function
    # is dependent on its node name, which could lead to a range of weird and unexpected behaviours.
    # A better way to implement the SEM would be to have a concept of directed graph instead of dict
    # where we can update the values in the order specified by the graph (not the name of the nodes).
    sample = {}
    for variable, function in model.items():
        sample[variable] = function(epsilon, **sample)
    return sample


def intervene_dict(model, **interventions):
    """Create a new SEM with fixed values for the variables of the base model that are intervened on.
    :type model: dict
    :param model: The SEM obtained with the define_sem method of any GraphStructure
    :type interventions: dict
    :param interventions: The interventions with format {node_name: intervention_value}
    :rtype: dict
    :return: The model after intervention
    """
    # TODO: This will need to be updated if the graph structure changes
    # The lambdas are kept for now because the original structure contains useful function,
    # the new model must thus be callable.
    # We use a copied model as only the variables intervened on will change.
    new_model = copy.deepcopy(model)
    new_model.update({k: lambda *args, **kwargs: v for k, v in interventions.items()})
    return new_model


def compute_interventions(model, interventions, node_values, target_variable="Y", num_samples=100000, seed=1):
    """Generate a new SEM where all the specified nodes are intervened on, and compute
    num_samples from this new SEM. Return the averaged values obtained on the target variable.
    :type model: dict
    :param model: The original SEM
    :type node_values:
    :param node_values:
    :type interventions: dict
    :param interventions: The nodes on which to intervene in the format {node_name: ""}
    :type target_variable: str
    :param target_variable: The name of the node corresponding to our target variable
    :type num_samples: int
    :param num_samples: The number of samples to generate
    :type seed: int
    :param seed: The seed used for sampling
    :rtype: np.array
    :return: The averaged values obtained on the target variable
    """
    # TODO: maybe change this to interventions = {node: node_value[i] for i, node in enumerate(interventions.keys())}
    # It is not clear whether the updated intervention dict will be used by another part of the code afterwards, so I
    # update the current version instead of using a comprehension list here.
    for i, node in enumerate(interventions.keys()):
        interventions[node] = node_values[0, i]

    mutilated_model = intervene_dict(model, **interventions)
    np.random.seed(seed)
    # TODO: it would probably be better to sample from the target variable only instead of sampling everything each
    #  time, especially for large graphs
    samples = pd.DataFrame([sample_from_model(mutilated_model) for _ in range(num_samples)])
    return np.asarray(np.mean(samples[target_variable]))[np.newaxis, np.newaxis]


def get_parameter_space(interventions, min_interventions, max_interventions):
    """ Instantiate the ParameterSpace of the interventions
    :type interventions: dict
    :param interventions: The nodes on which to intervene in the format {node_name: ""}
    :type min_interventions: list
    :param min_interventions: The minimum value for each node
    :type max_interventions: list
    :param max_interventions: The maximum value for each node
    :rtype: ParameterSpace
    :return: The parameter space of the intervention
    """
    return ParameterSpace([
        ContinuousParameter(*params) for params in zip(interventions.keys(), min_interventions, max_interventions)
    ])
