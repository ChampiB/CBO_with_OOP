import numpy as np
import scipy
import itertools
from numpy.random import randn


def update_hull(observational_samples, manipulative_variables):
    # This function computes the coverage of the observations
    list_variables = []

    for i in range(len(manipulative_variables)):
        list_variables.append(observational_samples[manipulative_variables[i]])

    stack_variables = np.transpose(np.vstack((list_variables)))
    coverage_obs = scipy.spatial.ConvexHull(stack_variables).volume

    return coverage_obs


def compute_coverage(observational_samples, manipulative_variables, dict_ranges):
    list_variables = []
    list_ranges = []

    for i in range(len(manipulative_variables)):
        list_variables.append(observational_samples[manipulative_variables[i]])
        list_ranges.append(dict_ranges[manipulative_variables[i]])

    vertices = list(itertools.product(*[list_ranges[i] for i in range(len(manipulative_variables))]))
    coverage_total = scipy.spatial.ConvexHull(vertices).volume

    stack_variables = np.transpose(np.vstack((list_variables)))
    coverage_obs = scipy.spatial.ConvexHull(stack_variables).volume
    hull_obs = scipy.spatial.ConvexHull(stack_variables)

    alpha_coverage = coverage_obs/coverage_total
    return alpha_coverage, hull_obs, coverage_total


def define_initial_data_cbo(interventional_data, num_interventions, exploration_set, task):

    objective = np.min if task == 'min' else np.max

    data_list = []
    data_x_list = []
    data_y_list = []
    opt_list = []

    for j in range(len(exploration_set)):
        data = interventional_data[j].copy()
        n_vars = data[0]
        data_x = np.asarray(data[(n_vars + 1)] if n_vars == 1 else data[(n_vars + 1):(n_vars * 2)][0])
        data_y = np.asarray(data[-1])

        if len(data_y.shape) == 1:
            data_y = data_y[:, np.newaxis]

        if len(data_x.shape) == 1:
            data_x = data_x[:, np.newaxis]
      
        all_data = np.concatenate((data_x, data_y), axis=1)

        state = np.random.get_state()
        np.random.shuffle(all_data)
        np.random.set_state(state)

        subset_all_data = all_data[:num_interventions]

        data_list.append(subset_all_data)
        data_x_list.append(data_list[j][:, :-1])
        data_y_list.append(data_list[j][:, -1][:, np.newaxis])

        opt_list.append(objective(subset_all_data[:, -1]))
        opt_y = objective(opt_list)
        var_min = exploration_set[np.where(opt_list == opt_y)[0][0]]
        opt_intervention = data_list[np.where(opt_list == opt_y)[0][0]]

    shape_opt = opt_intervention.shape[1] - 1
    best = objective(opt_intervention[:, shape_opt])
    best_intervention_value = opt_intervention[opt_intervention[:, shape_opt] == best, :shape_opt][0]

    return data_x_list, data_y_list, best_intervention_value, opt_y, sum(var_min)
