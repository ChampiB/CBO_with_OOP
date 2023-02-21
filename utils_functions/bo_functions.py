import numpy as np
from numpy.random import randn
from GPy.core import Mapping
from GPy.kern import RBF
from GPy.models.gp_regression import GPRegression
from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from .causal_kernels import CausalRBF


def define_initial_data_bo(interventional_data, num_interventions, intervention_sets, name_index):
    data_x = (interventional_data[0][len(intervention_sets)]).copy()
    data_y = (np.asarray(interventional_data[0][len(intervention_sets)+1])).copy()
    all_data = np.concatenate((data_x, data_y), axis=1)

    # Need to reset the global seed
    state = np.random.get_state()

    np.random.seed(name_index)
    np.random.shuffle(all_data)
    
    np.random.set_state(state)

    data_x = all_data[:num_interventions, :len(intervention_sets)]
    data_y = all_data[:num_interventions, len(intervention_sets):]

    min_y = np.min(data_y)
    min_intervention_value = np.transpose(all_data[np.where(data_y == min_y)[0][0]][:len(intervention_sets)][:, np.newaxis])

    return data_x, data_y, min_intervention_value, min_y


def update_gaussian_processes(mean_function, var_function, data_x, data_y, causal_prior):
    # This function updates the BO model for each intervention set
    if causal_prior is False:
        gpy_model = GPRegression(
            data_x, data_y, RBF(data_x.shape[1], lengthscale=1., variance=1.),  noise_var=1e-10
        )
    else:
        mf = Mapping(data_x.shape[1], 1)
        mf.f = lambda x: mean_function(x)
        mf.update_gradients = lambda a, b: None
        causal_kernel = CausalRBF(
            data_x.shape[1], variance_adjustment=var_function, lengthscale=1., variance=1., ARD=False
        )
        gpy_model = GPRegression(data_x, data_y, causal_kernel, noise_var=1e-10, mean_function=mf)

    # Wrap the GPy model to make it compatible with the emukit package
    return GPyModelWrapper(gpy_model)
