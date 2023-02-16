# Import basic packages
import numpy as np
import pandas as pd
from collections import OrderedDict
import scipy
import itertools
from numpy.random import randn
import copy
import seaborn as sns


def get_saving_dir(experiment, type_cost, initial_num_obs_samples, num_interventions):
    cost_types = ["fix_equal", "fix_different", "fix_different_variable", "fix_equal_variable"]
    return f"./data/{experiment}/{cost_types[type_cost]}/{initial_num_obs_samples}/{num_interventions}/"


def save_results_bayesian_optimisation(folder, args, current_cost, current_best_x, current_best_y, total_time, Causal_prior):
    np.save(folder + "cost_" + str(args.exploration_set) + '_' + str(Causal_prior) + str(args.name_index) + ".npy", current_cost)
    np.save(folder + "best_x_" + str(args.exploration_set) + '_' + str(Causal_prior) + str(args.name_index) + ".npy",current_best_x)
    np.save(folder + "best_y_" + str(args.exploration_set) + '_' + str(Causal_prior) + str(args.name_index) + ".npy", current_best_y)
    np.save(folder + "total_time_" + str(args.exploration_set) + '_' + str(Causal_prior) + str(args.name_index) + ".npy",total_time)
