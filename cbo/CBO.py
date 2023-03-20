from pathlib import Path
from cbo.DoCalculus import DoCalculus
from cbo.Monitor import Monitor
from cbo.utils_functions.cbo_functions import *
from cbo.utils_functions.utils import *
from cbo.utils_functions.cost_functions import *
from numpy.random import uniform
from cbo.GaussianProcessFactory import GaussianProcessFactory as GPFactory
from cbo.GaussianProcessFactory import GaussianProcessType as GPType


class CBO:
	"""
	A class implementing the Causal Bayesian Optimisation agent.
	"""

	def __init__(self, config, graph, verbose=True):
		"""
		Create the CBO agent
		:param config: the hydra configuration
		:param graph: the graph on which CBO should be run
		:param verbose: whether to display debug information
		"""

		# Store the loaded data.
		self.graph = graph
		self.measurements = graph.measurements
		self.all_measurements = graph.all_measurements
		self.interventions = graph.interventions

		# Store useful arguments.
		self.exploration_set = self.graph.exploration_set
		self.es_size = len(self.exploration_set)
		self.num_interventions = config.n_interventions
		self.max_n = config.n_initial_samples + 50
		self.n_initial_samples = config.n_initial_samples
		self.gp_type = GPType.CAUSAL_GP if config.causal_prior else GPType.NON_CAUSAL_GP
		self.num_trials = config.n_trials
		self.task = config.task
		self.n_new_observations = config.n_new_observations
		self.cost_type = config.cost_type

		# Set the seed for reproducibility
		np.random.seed(config.seed)

		# List that will contain the mean and variance functions of the Gaussian processes.
		self.mean_functions = []
		self.var_functions = []

		# List that will contain the Gaussian processes.
		self.models = []

		# Get the cost corresponding to the `type_cost` passed as arguments.
		self.costs = self.graph.get_cost_structure(type_cost=self.cost_type)

		# Get the path to the saving directory, and create it if it does not exist.
		self.saving_dir = self.get_saving_dir(graph.name, self.num_interventions)
		Path(self.saving_dir).mkdir(parents=True, exist_ok=True)

		# Get the interventions' name for each intervention in the exploration_set
		self.intervention_names = ["".join(variables) for variables in self.exploration_set]

		# For each Gaussian process, initialise the mean and variance of x
		self.x_mean = {i: {} for i in self.intervention_names}
		self.x_var = {i: {} for i in self.intervention_names}

		# Create the monitor that will keep track of the CBO performance.
		self.monitor = Monitor(self, verbose=verbose)

		# Create the do-calculus engine.
		self.do_calculus = DoCalculus(self)

		# Store verbose mode.
		self.verbose = verbose

	def get_saving_dir(self, experiment, num_interventions):
		"""
		Getter
		:param experiment: the experiment to run
		:param num_interventions: the number of interventions
		:return: the path to the saving directory
		"""
		cost_types = ["fix_equal", "fix_different", "fix_different_variable", "fix_equal_variable"]
		return f"./data/{experiment}/{cost_types[self.cost_type]}/{self.n_initial_samples}/{num_interventions}/"

	def run(self):
		"""
		Run the Causal Bayesian Optimisation algorithm
		"""

		# Display information about the experiment being run, if requested by the user.
		if self.verbose is True:
			print(f"Exploring {self.exploration_set} with CEO and Causal prior = {self.gp_type}")

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
			print('causal_prior: ', self.gp_type)
			print('type_cost: ', self.cost_type)
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
		gaussian_processes = self.graph.fit_all_gaussian_processes(self.measurements)

		# Update the mean and variance functions to account for the new observational data.
		self.mean_functions, self.var_functions = self.do_calculus.update_all_do_functions(gaussian_processes)

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
		coverage_total = compute_coverage(self.measurements, self.graph.manipulative_variables, self.interventional_ranges)[2]

		# Compute epsilon.
		coverage_obs = update_hull(self.measurements, self.graph.manipulative_variables)
		rescale = self.measurements.shape[0] / self.max_n
		return (coverage_obs / coverage_total) / rescale

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
			num_observation=self.n_new_observations,
			complete_dataset=self.all_measurements,
			initial_num_obs_samples=self.n_initial_samples
		)

	def update_all_gaussian_processes(self):
		"""
		Update all the Gaussian processes using the newly available data
		"""
		self.models = [
			GPFactory.create(
				self.gp_type,
				self.monitor.data_x[s],
				self.monitor.data_y[s],
				[self.mean_functions[s], self.var_functions[s]],
				emukit_wrapper=True
			)
			for s in range(self.es_size)
		]

	def update_gaussian_process_of_last_intervention(self):
		"""
		Update only the Gaussian process corresponding to the last intervention performed
		"""
		last_intervention = self.monitor.last_intervention
		self.models[last_intervention] = GPFactory.create(
			self.gp_type,
			self.monitor.data_x[last_intervention],
			self.monitor.data_y[last_intervention],
			[self.mean_functions[last_intervention], self.var_functions[last_intervention]],
			emukit_wrapper=True
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
		return find_current_global(self.monitor.current_best_y, self.intervention_names, self.task)

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
		x = {
			intervention_var: acquisition_xs[intervention][0, i]
			for i, intervention_var in enumerate(intervention_set)
		}
		return total_cost(intervention_set, self.costs, x)
