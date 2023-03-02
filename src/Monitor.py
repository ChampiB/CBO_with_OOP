import time
from src.utils_functions import *
import copy


class Monitor:
    """
    A class used to monitor the CBO agent.
    """

    def __init__(self, cbo, verbose=False):
        """
        Create an object used to monitor the CBO agent
        :param cbo: the CBO agent to monitor
        :param verbose: whether to display debug information
        """

        # Store the CBO agent and the verbose mode
        self.cbo = cbo
        self.verbose = verbose

        # Get the initial optimal solution and the interventional data corresponding to a random permutation of the
        # interventional data with seed given by name_index
        self.data_x, self.data_y, best_intervention_value, opt_y, best_variable = \
            define_initial_data_cbo(
                cbo.interventions, cbo.num_interventions, eval(cbo.exploration_set), cbo.name_index, cbo.task
            )
        self.current_cost = [0.]
        self.global_opt = [opt_y]

        # For each Gaussian process, initialise the x-position that leads to the best acquisition value
        self.current_best_x = {i: [np.inf if cbo.task == 'min' else -np.inf] for i in self.cbo.interventions}
        self.current_best_y = copy.deepcopy(self.current_best_x)
        self.current_best_y[best_variable].append(opt_y)
        self.current_best_x[best_variable].append(best_intervention_value)

        # Store important metrics
        self.observed = 0
        self.trial_intervened = 0.
        self.cumulative_cost = 0.

        # Define list to store info
        self.target_function_list = []
        self.space_list = []
        self.type_trial = []

        # Define intervention function
        for s in range(len(cbo.exploration_set)):
            ranges = cbo.graph.get_interventional_ranges()
            min_ranges = [ranges[intervention][0] for intervention in cbo.exploration_set[s]]
            max_ranges = [ranges[intervention][1] for intervention in cbo.exploration_set[s]]
            target_function, space = intervention_function(
                {intervention: '' for intervention in cbo.exploration_set[s]},
                model=cbo.graph.define_sem(),
                target_variable='Y',
                min_intervention=min_ranges,
                max_intervention=max_ranges
            )
            self.target_function_list.append(target_function)
            self.space_list.append(space)

        # Counter keeping track of the trial number
        self.i = 0

        # The index of the last intervention set on which we intervened.
        self.last_intervention = None

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

    def log_agent_performance(self, intervention_set=None, intervention=None, acquisition_xs=None, current_cost=None):
        """
        Log metrics representing the agent's performance
        :param intervention_set: the set of variables on which the agent intervene
        :param intervention: the values of the variables in the intervention_set
        :param acquisition_xs: the values of x that produces the highest acquisition values
        :param current_cost: the cost of performing the intervention
        """

        # If the agent observes, then the cost and optimal reward stay the same as in the previous trial.
        if current_cost is None:
            self.global_opt.append(self.global_opt[-1])
            self.current_cost.append(self.current_cost[-1])
            return

        # Evaluate the target function for the intervention performed.
        target_ys = self.compute_target_function(intervention_set, intervention, acquisition_xs)

        # Add the new data to the dataset used by CBO.
        self.add_intervention_data(target_ys, intervention, acquisition_xs)

        # Update the dict storing the current optimal solution.
        var_to_intervene = self.cbo.interventions[intervention]
        self.current_best_x[var_to_intervene].append(acquisition_xs[intervention][0][0])
        self.current_best_y[var_to_intervene].append(target_ys[0][0])

        # Find the new current best solution.
        current_best = find_current_global(self.current_best_y, self.cbo.interventions, self.cbo.task)

        # Otherwise, the cost and optimal reward are provided as parameters.
        self.global_opt.append(current_best)
        self.cumulative_cost += current_cost
        self.current_cost.append(self.cumulative_cost)

        # Display optimal reward, if requested by the user.
        if self.verbose is True:
            print('####### Current_global #########', current_best)

    def agent_previously_observed(self):
        """
        Getter
        :return: True, if the agent observed during the previous trial, False otherwise
        """
        return self.type_trial[-2] == 0

    def add_intervention_data(self, target_ys, intervention, acquisition_xs):
        """
        Add a new entry in the dataset
        :param target_ys: the target values
        :param intervention: the intervention perform
        :param acquisition_xs: the values of x that produces the highest acquisition values
        """
        data_x = np.append(self.data_x[intervention], acquisition_xs[intervention], axis=0)
        data_y = np.append(self.data_y[intervention], target_ys, axis=0)

        self.data_x[intervention] = np.vstack((self.data_x[intervention], acquisition_xs[intervention]))
        self.data_y[intervention] = np.vstack((self.data_y[intervention], target_ys))
        self.cbo.models[intervention].set_data(data_x, data_y)

    def compute_target_function(self, intervention_set, intervention, acquisition_xs):
        """
        Compute the target function
        :param intervention_set: the set of variables on which the agent intervene
        :param intervention: the intervention perform
        :param acquisition_xs: the values of x that produces the highest acquisition values
        :return: the target values
        """

        # Compute the value of the target function
        y_new = self.target_function_list[intervention](acquisition_xs[intervention])

        # Display debug information, if requested by the user.
        if self.verbose is True:
            print('Selected intervention set: ', intervention_set)
            print('Selected values: ', acquisition_xs[intervention])
            print('Target function at the selected values: ', y_new)
        return y_new

    def save_results(self):
        """
        Save the results of the monitored CBO agent
        """
        index = f"{self.cbo.exploration_set}_{self.cbo.gp_type}_{self.cbo.name_index}"
        np.save(self.cbo.saving_dir + f"cost_{index}.npy", self.current_cost)
        np.save(self.cbo.saving_dir + f"best_x_{index}.npy", self.current_best_x)
        np.save(self.cbo.saving_dir + f"best_y_{index}.npy", self.current_best_y)
        np.save(self.cbo.saving_dir + f"total_time_{index}.npy", self.total_time)
        np.save(self.cbo.saving_dir + f"observed_{index}.npy", self.observed)
        np.save(self.cbo.saving_dir + f"global_opt_{index}.npy", self.global_opt)
