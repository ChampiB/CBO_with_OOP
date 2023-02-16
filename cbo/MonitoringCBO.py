import time
from utils_functions import *


class MonitoringCBO:
    """
    A class used to monitor the CBO agent.
    """

    def __init__(self, cbo, verbose=False):
        # TODO Refacto this function

        self.cbo = cbo
        self.verbose = verbose

        # Get the initial optimal solution and the interventional data corresponding to a random permutation of the
        # interventional data with seed given by name_index
        self.data_x_list, self.data_y_list, best_intervention_value, opt_y, best_variable = \
            define_initial_data_CBO(
                cbo.interventions, cbo.num_interventions, eval(cbo.exploration_set), cbo.name_index, cbo.task
            )
        self.current_cost = [0.]
        self.global_opt = [opt_y]
        self.current_best_x, self.current_best_y, self.x_dict_mean, self.x_dict_var, self.dict_interventions = \
            initialise_dicts(cbo.exploration_set, cbo.task)
        self.current_best_y[best_variable].append(opt_y)
        self.current_best_x[best_variable].append(best_intervention_value)

        self.observed = 0
        self.trial_intervened = 0.
        self.cumulative_cost = 0.

        # Define list to store info
        n_exploration_sets = len(cbo.exploration_set)
        self.target_function_list = [None] * n_exploration_sets
        self.space_list = [None] * n_exploration_sets
        self.model_list = [None] * n_exploration_sets
        self.type_trial = []

        # Define intervention function
        for s in range(len(cbo.exploration_set)):
            interventions = list_interventional_ranges(cbo.graph.get_interventional_ranges(), cbo.exploration_set[s])
            self.target_function_list[s], self.space_list[s] = Intervention_function(
                get_interventional_dict(cbo.exploration_set[s]),
                model=cbo.graph.define_SEM(),
                target_variable='Y',
                min_intervention=interventions[0],
                max_intervention=interventions[1]
            )

        self.i = 0

        # Time tracking attributes
        self.start_time = None
        self.total_time = None

    def start(self):
        """
        Start monitoring the CBO agent
        """
        self.start_time = time.clock()

    def stop(self):
        """
        Stop monitoring the CBO agent
        """
        self.total_time = time.clock() - self.start_time

    def log_agent_behaviour(self, act):
        """
        Log the agent behaviour, i.e., whether the agent performed an intervention or made an observation
        :param act: True, if the agent is acting, False otherwise (it is observing)
        """
        # Display the current optimisation step.
        if self.verbose is True:
            print('Optimization step', self.i)

        # Keep track of when the agent intervened and observed.
        if act is True:
            self.type_trial.append(1)
            self.trial_intervened += 1
        else:
            self.observed += 1
            self.type_trial.append(0)

        # Increase the optimisation step.
        self.i += 1

    def log_agent_performance(self, global_opt=None, current_cost=None):
        # If the agent observes, then the cost and optimal reward stay the same as in the previous trial.
        if global_opt is None and current_cost is None:
            self.global_opt.append(self.global_opt[-1])
            self.current_cost.append(self.current_cost[-1])
        else:
            pass  # TODO

    def save_results(self):
        """
        Save the results of the monitored CBO agent
        """
        index = f"{self.cbo.exploration_set}_{self.cbo.causal_prior}_{self.cbo.name_index}"
        np.save(self.cbo.saving_dir + f"cost_{index}.npy", self.current_cost)
        np.save(self.cbo.saving_dir + f"best_x_{index}.npy", self.current_best_x)
        np.save(self.cbo.saving_dir + f"best_y_{index}.npy", self.current_best_y)
        np.save(self.cbo.saving_dir + f"total_time_{index}.npy", self.total_time)
        np.save(self.cbo.saving_dir + f"observed_{index}.npy", self.observed)
        np.save(self.cbo.saving_dir + f"global_opt_{index}.npy", self.global_opt)
