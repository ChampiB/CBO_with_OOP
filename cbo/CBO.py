from pathlib import Path
from cbo.MonitoringCBO import MonitoringCBO
from utils_functions import *
from numpy.random import uniform


# TODO self.graph.fit_all_gaussian_processes() => fit a Gaussian process using GPy

# TODO self.update_all_do_functions(functions) => compute all mean and var functions using do-calculus, how?

# TODO self.update_all_gaussian_processes() => Create new Gaussian processes model, i.e., no fitting
# TODO self.update_gaussian_process_of_last_intervention() => Create a new Gaussian process model, i.e., no fitting

# TODO self.model_list[intervention].optimize() => fit the Gaussian process

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

		# List that will contain the mean and variance functions of the Gaussian processes.
		self.mean_functions = []
		self.var_functions = []

		# List that will contain the Gaussian processes.
		self.models = []

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
		self.graph.fit_all_gaussian_processes()

		# Make sure the agent observes and intervenes at least once.
		self.observe()
		self.intervene()

		# Perform the requested number of trials.
		self.monitor.start()
		for i in range(self.num_trials - 2):
			# Perform the epsilon-greedy policy, the agent observes if uniform < epsilon otherwise it intervenes.
			if uniform(0., 1.) < self.epsilon:
				self.observe()
			else:
				self.intervene()
		self.monitor.stop()

		# Save the monitoring results.
		self.monitor.save_results()

		# Display the saved results, if requested.
		if self.verbose is True:
			print('=================================== Saved results ===================================')
			print('exploration_set: ', self.exploration_set)
			print('causal_prior: ', self.causal_prior)
			print('type_cost: ', self.type_cost)
			print('total_time: ', self.monitor.total_time)
			print('folder: ', self.saving_dir)
			print('=====================================================================================')
			print()

	def observe(self):
		"""
		Implementation of the agent's behaviour when it observes.
		"""

		# Track that the agent is observing.
		self.monitor.log_agent_behaviour(act=False)

		# Collect a new observation, and add it to the observational dataset.
		self.measurements = self.measurements.append(self.get_new_observation())

		# Fit the models using the newly available data.
		functions = self.graph.fit_all_gaussian_processes(self.measurements)

		# Update the mean and variance functions to account for the new observational data.
		self.mean_functions, self.var_functions = self.update_all_do_functions(functions)

		# Log the cost and optimal reward (i.e., as the agent is observing, it remains the same previous trial).
		self.monitor.log_agent_performance()

	def intervene(self):
		"""
		Implementation of the agent's behaviour when it intervenes.
		"""

		# Track that the agent is acting.
		self.monitor.log_agent_behaviour(act=True)

		# Find the best solution found by CBO so far.
		current_best = self.current_best_solution()

		# Update either all the Gaussian processes or only the one corresponding to the last intervention.
		if self.monitor.agent_previously_observed():
			self.update_all_gaussian_processes()
		else:
			self.update_gaussian_process_of_last_intervention()

		# Compute the highest values taken by each acquisition function, and retrieve the associated best values of x.
		acquisition_xs, acquisition_ys = self.compute_best_acquisition_values(current_best)

		# Select the variables on which to intervene and their values, based on the best acquisition values.
		intervention_set, intervention = self.select_next_intervention(acquisition_ys)

		# Compute the cost of the performed intervention.
		current_cost = self.compute_cost(intervention_set, intervention, acquisition_xs)

		# Log the agent performance.
		self.monitor.log_agent_performance(intervention_set, intervention, acquisition_xs, current_cost)

		# Optimise the Gaussian process corresponding to the intervention performed.
		self.models[intervention].optimize()

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

		# Create the lists of mean and variance functions.
		mean_functions = []
		var_functions = []

		# Update the mean and variance functions.
		for j in range(len(self.exploration_set)):
			mean_functions.append(update_mean_fun(
				self.graph, functions, self.monitor.dict_interventions[j], self.measurements, self.monitor.x_dict_mean
			))
			var_functions.append(update_var_fun(
				self.graph, functions, self.monitor.dict_interventions[j], self.measurements, self.monitor.x_dict_var
			))
		return mean_functions, var_functions

	def update_all_gaussian_processes(self):
		"""
		Update all the Gaussian processes using the newly available data
		"""
		self.models.clear()
		for s in range(len(self.exploration_set)):
			model = update_gaussian_processes(
				self.mean_functions[s],
				self.var_functions[s],
				self.monitor.data_x_list[s],
				self.monitor.data_y_list[s],
				self.causal_prior
			)
			self.models.append(model)

	def update_gaussian_process_of_last_intervention(self):
		"""
		Update only the Gaussian process corresponding to the last intervention performed
		"""
		last_intervention = self.monitor.last_intervention
		self.models[last_intervention] = update_gaussian_processes(
			self.mean_functions[last_intervention],
			self.var_functions[last_intervention],
			self.monitor.data_x_list[last_intervention],
			self.monitor.data_y_list[last_intervention],
			self.causal_prior
		)

	def compute_best_acquisition_values(self, current_best):
		"""
		Compute the highest value taken by each acquisition function, i.e., y = f(x), and the associated values of x
		:param current_best: the current best solution
		:return: the xs and ys
		"""

		# Define the lists storing the optimal values of the acquisition functions, and the associated x-positions.
		xs = []
		ys = []

		# Compute the highest acquisition values, i.e., y* = f(x*), and the associated best values of x.
		for s in range(len(self.exploration_set)):
			y, x = find_next_y_point(
				self.monitor.space_list[s],
				self.models[s],
				current_best,
				self.exploration_set[s],
				self.costs,
				task=self.task
			)
			ys.append(y)
			xs.append(x)
		return xs, ys

	def current_best_solution(self):
		"""
		Find the current best solution found by CBO
		:return: the current best solution
		"""
		return find_current_global(self.monitor.current_best_y, self.monitor.dict_interventions, self.task)

	def select_next_intervention(self, acquisition_ys):
		"""
		Select the next intervention to be performed based on the value of the acquisition functions
		:param acquisition_ys: the highest values taken by the acquisition function
		:return: the index of the set on which to intervene, and the values taken by the variables in that set
		"""
		indices = np.where(acquisition_ys == np.max(acquisition_ys))[0][0]
		self.monitor.last_intervention = np.where(acquisition_ys == np.max(acquisition_ys))[0][0]
		return self.exploration_set[indices], self.monitor.last_intervention

	def compute_cost(self, intervention_set, intervention, acquisition_xs):
		"""
		Compute cost of performing an intervention
		:param intervention_set: the set of variables on which the intervention is performed
		:param intervention: the intervention
		:param acquisition_xs: the values of x that maximise the acquisition function
		:return: the intervention's cost
		"""
		x = get_new_dict_x(acquisition_xs[intervention], self.monitor.dict_interventions[intervention])
		return total_cost(intervention_set, self.costs, x)
