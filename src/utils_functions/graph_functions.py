import numpy as np
import pandas as pd
from numpy.random import randn
from emukit.core import ParameterSpace, ContinuousParameter


def sample_from_model(model, epsilon = None):
    # Produces a single sample from a structural equation model.
    if epsilon is None:
        epsilon = randn(len(model))
    sample = {}
    for variable, function in model.items():
        sample[variable] = function(epsilon, **sample)
    return sample


def intervene_dict(model, **interventions):

    new_model = model.copy()

    def assign(model, variable, value):
        model[variable] = lambda epsilon, **kwargs : value
        
    for variable, value in interventions.items():
        assign(new_model, variable, value)
  
    return new_model


def intervention_function(*interventions, model, target_variable, min_intervention, max_intervention):
    num_samples = 100000

    assert len(min_intervention) == len(interventions[0])
    assert len(max_intervention) == len(interventions[0])

    def compute_target_function_fcn(value):
        num_interventions = len(interventions[0])
        for i in range(num_interventions):
            interventions[0][list(interventions[0].keys())[i]] = value[0,i]
    
        mutilated_model = intervene_dict(model, **interventions[0])
        np.random.seed(1)
        samples = [sample_from_model(mutilated_model) for _ in range(num_samples)]
        samples = pd.DataFrame(samples)
        return np.asarray(np.mean(samples['Y']))[np.newaxis,np.newaxis]
    
    # Define parameter space
    list_parameter = [None] * len(interventions[0])
    for i in range(len(interventions[0])):
        list_parameter[i] = ContinuousParameter(
            list(interventions[0].keys())[i],
            min_intervention[i], max_intervention[i]
        )
        
    return (compute_target_function_fcn, ParameterSpace(list_parameter))
