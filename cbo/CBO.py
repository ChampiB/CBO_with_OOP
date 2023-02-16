from pathlib import Path
from cbo.MonitoringCBO import MonitoringCBO
from utils_functions import *


class CBO:
	"""
	A class implementing the Causal Bayesian Optimisation agent.
	"""

	def __init__(self, args, data, verbose=True):
		"""
		Create the CBO agent
		:param args: the scripts arguments
		:param data: the data loader
		:param verbose: whether to display debug information
		"""

		# Store useful arguments.
		self.max_n = args.initial_num_obs_samples + 50
		self.initial_num_obs_samples = args.initial_num_obs_samples
		self.num_interventions = args.num_interventions
		self.causal_prior = args.causal_prior
		self.num_trials = args.num_trials
		self.exploration_set = eval(args.exploration_set)
		self.task = args.task
		self.num_additional_observations = args.num_additional_observations
		self.type_cost = args.type_cost
		self.name_index = args.name_index

		# Store the loaded data.
		self.graph = data.graph
		self.measurements = data.measurements
		self.all_measurements = data.all_measurements
		self.interventions = data.interventions

		# Get the cost corresponding to the `type_cost` passed as arguments.
		self.costs = self.graph.get_cost_structure(type_cost=self.type_cost)

		# Get the path to the saving directory, and create it if it does not exist.
		self.saving_dir = \
			get_saving_dir(args.experiment, args.type_cost, args.initial_num_obs_samples, args.num_interventions)
		Path(self.saving_dir).mkdir(parents=True, exist_ok=True)

		# Create the monitor that will keep track of the CBO performance.
		self.monitor = MonitoringCBO(self, verbose=verbose)

		# Store verbose mode.
		self.verbose = verbose

	def run(self):
		"""
		Run the Causal Bayesian Optimisation algorithm
		"""

		# Display information about the experiment being run, if requested by the user.
		if self.verbose is True:
			print(f"Exploring {self.exploration_set} with CEO and Causal prior = {self.causal_prior}")

		# Fit all the Gaussian processes using the available data.
		self.graph.fit_all_models()

		# Perform the requested number of trials.
		self.monitor.start()
		for i in range(self.num_trials):

			# Perform epsilon-greedy policy.
			if self.sample_uniform(i) < self.epsilon:

				# Track that the agent is observing.
				self.monitor.log_agent_behaviour(act=False)

				# Collect a new observation, and add it to the observational dataset.
				self.measurements = self.measurements.append(self.get_new_observation())

				# Fit the models using the newly available data.
				functions = self.graph.refit_models(self.measurements)

				# Update the mean and variance functions to account for the new observational data.
				mean_functions_list, var_functions_list = self.update_all_do_functions(functions)

				# Log the cost and optimal reward (i.e., as the agent is observing, it remains the same previous trial).
				self.monitor.log_agent_performance()
			else:

				# Track that the agent is acting
				self.monitor.log_agent_behaviour(act=True)

				# When we decide to intervene we need to compute the acquisition functions based on the GP models and
				# decide the variable/variables to intervene together with their interventional data

				# Define list to store info
				y_acquisition_list = [None] * len(self.exploration_set)
				x_new_list = [None] * len(self.exploration_set)

				# This is the global opt from previous iteration
				current_global = find_current_global(self.monitor.current_best_y, self.monitor.dict_interventions, self.task)

				# If in the previous trial we have observed we want to update all the BO models as the mean functions
				# and var functions computed via the DO calculus are changed.
				# If in the previous trial we have intervened we want to update only the BO model for the intervention
				# for which we have collected additional data
				if self.monitor.type_trial[i-1] == 0:
					for s in range(len(self.exploration_set)):
						self.monitor.model_list[s] = update_BO_models(
							mean_functions_list[s], var_functions_list[s],
							self.monitor.data_x_list[s], self.monitor.data_y_list[s], self.causal_prior
						)
				else:
					self.monitor.model_list[index] = update_BO_models(
						mean_functions_list[index],
						var_functions_list[index],
						self.monitor.data_x_list[index], self.monitor.data_y_list[index], self.causal_prior
					)

				# Compute acquisition function given the updated BO models for the interventional data
				# Notice that we use current_global and the costs to compute the acquisition functions
				for s in range(len(self.exploration_set)):
					y_acquisition_list[s], x_new_list[s] = \
						find_next_y_point(
							self.monitor.space_list[s], self.monitor.model_list[s], current_global,
							self.exploration_set[s], self.costs, task=self.task
						)

				# Selecting the variable to intervene based on the values of the acquisition functions
				var_to_intervene = self.exploration_set[np.where(y_acquisition_list == np.max(y_acquisition_list))[0][0]]
				index = np.where(y_acquisition_list == np.max(y_acquisition_list))[0][0]

				# Evaluate the target function at the new point
				y_new = self.monitor.target_function_list[index](x_new_list[index])

				print('Selected intervention: ', var_to_intervene)
				print('Selected point: ', x_new_list[index])
				print('Target function at selected point: ', y_new)

				# Append the new data and set the new dataset of the BO model
				data_x, data_y_x = add_data(
					[self.monitor.data_x_list[index], self.monitor.data_y_list[index]], [x_new_list[index], y_new]
				)

				self.monitor.data_x_list[index] = np.vstack((self.monitor.data_x_list[index], x_new_list[index]))
				self.monitor.data_y_list[index] = np.vstack((self.monitor.data_y_list[index], y_new))

				self.monitor.model_list[index].set_data(data_x, data_y_x)

				# Compute cost
				x_new_dict = get_new_dict_x(x_new_list[index], self.monitor.dict_interventions[index])
				self.monitor.cumulative_cost += total_cost(var_to_intervene, self.costs, x_new_dict)
				var_to_intervene = self.monitor.dict_interventions[index]
				self.monitor.current_cost.append(self.monitor.cumulative_cost)

				# Update the dict storing the current optimal solution
				self.monitor.current_best_x[var_to_intervene].append(x_new_list[index][0][0])
				self.monitor.current_best_y[var_to_intervene].append(y_new[0][0])

				# Find the new current global optima
				current_global = find_current_global(self.monitor.current_best_y, self.monitor.dict_interventions, self.task)
				self.monitor.global_opt.append(current_global)

				print('####### Current_global #########', current_global)

				# Optimise BO model given the new data
				self.monitor.model_list[index].optimize()

		# Stop monitoring the CBO agent and save the monitoring results
		self.monitor.stop()
		self.monitor.save_results()

		# Display the saved results, if requested
		if self.verbose is True:
			print('=================================== Saved results ===================================')
			print('')
			print('exploration_set', self.exploration_set)
			print('causal_prior', self.causal_prior)
			print('type_cost', self.type_cost)
			print('total_time', self.monitor.total_time)
			print('folder', self.saving_dir)
			print('=====================================================================================')
			print()

	@staticmethod
	def sample_uniform(i):
		"""
		Sample a number from a uniform distribution, except if i == 0, in which case the agent will observe, and
		if i == 1, in which case the agent will intervene
		:return: a number sampled from a uniform distribution, except if i < 2
		"""

		# Observe at least once.
		if i == 0:
			return 0.

		# Intervene at least once.
		if i == 1:
			return 1.

		# Sample from the
		return np.random.uniform(0., 1.)

	@property
	def epsilon(self):
		"""
		Getter
		:return: the value of epsilon based on the available measurements
		"""

		# Compute the observation coverage.
		coverage_total = compute_coverage(self.measurements, self.manipulative_variables, self.interventional_ranges)[2]

		# Compute epsilon.
		coverage_obs = update_hull(self.measurements, self.manipulative_variables)
		rescale = self.measurements.shape[0] / self.max_n
		return (coverage_obs / coverage_total) / rescale

	@property
	def manipulative_variables(self):
		"""
		Getter
		:return: the manipulative variables
		"""
		return self.graph.get_sets()[2]

	@property
	def interventional_ranges(self):
		"""
		Getter
		:return: the interventional ranges
		"""
		return self.graph.get_interventional_ranges()

	def get_new_observation(self):
		"""
		Retrieve a new observation
		:return: the new observation
		"""
		return observe(
			num_observation=self.num_additional_observations,
			complete_dataset=self.all_measurements,
			initial_num_obs_samples=self.initial_num_obs_samples
		)

	def update_all_do_functions(self, functions):
		"""
		Compute the new mean and variance functions
		:param functions: the previous functions
		:return: the new mean and variance functions
		"""
		mean_functions_list = []
		var_functions_list = []

		for j in range(len(self.exploration_set)):
			mean_functions_list.append(update_mean_fun(
				self.graph, functions, self.monitor.dict_interventions[j], self.measurements, self.monitor.x_dict_mean
			))
			var_functions_list.append(update_var_fun(
				self.graph, functions, self.monitor.dict_interventions[j], self.measurements, self.monitor.x_dict_var
			))
		return mean_functions_list, var_functions_list
