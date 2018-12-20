import os
import numpy as np
from keras import Model
from keras.callbacks import TensorBoard
from numpy import ndarray
from time import time
from random import random, randrange
from configparser import ConfigParser
from Util.stats import DistributionInfo as Stat


class QTrainer:
	DEFAULT_SAVE_PATH = "./NeuralNetworks/saved/"
	DEFAULT_CONFIG_FILE = "./NeuralNetworks/default_q_config.cfg"
	DEFAULT_TEMP_DIR = "./NeuralNetworks/temp/"

	def __init__(self, model: Model, config: QTrainerConfig = None, config_file: str = DEFAULT_CONFIG_FILE):
		# configuration
		if config is None:
			config = QTrainerConfig(config_file)
		self.cfg = config
		self.tb_callback = TensorBoard(log_dir=self.cfg["save_path"] + self.cfg["name"] + "/tb/", histogram_freq=1)

		self.model = model
		self.replay_memory = ReplayMemory(self.cfg.get_("discount_factor"))
		self.epsilon = self.cfg.get_("epsilon")

		# information
		self.outputs = []
		self.classifications = []
		self.iter_len_secs = []
		self.rewards = []
		self.ep_len_iters = []
		self.ep_len_secs = []

		self.iteration_count = 0
		self.last_iteration = None
		self.last_episode = None
		self.inter_iter_save = None

	def iteration(self, state: ndarray, verbose: bool = False) -> list:
		"""
		perform a singe iteration of calculating an action from the state
		:param state: the state of the environment (must be provided as tensor/numpy array of type float)
		:param verbose: if true additional information about the process is printed to the console
		:return: the chosen output/classification
		"""
		# getting the results from the net
		output = self.model.predict(state, batch_size=1 if verbose else 0)
		self.outputs.append(output)

		classification = np.argmax(output)
		self.classifications.append(classification)

		# with a chance of epsilon a random classification is chosen instead
		if random() < self.cfg.get_("epsilon"):
			classification = randrange(0, len(classification))

		if self.inter_iter_save is not None:
			print("!Warning: no reward was recorded in the last iteration")
		self.inter_iter_save = (state, output, classification)

		# time measurement
		self.iteration_count += 1
		cur_time = time()
		if self.last_iteration is not None:
			self.iter_len_secs.append(cur_time - self.last_iteration)
		self.last_iteration = cur_time

		return classification

	def reward(self, reward: float) -> None:
		"""
		Because reward can only be given after the output is returned, it is
		separated from the rest of the iteration
		:param reward: the reward given in the last iteration
		:return: None
		"""
		self.rewards.append(reward)
		state, q_values, classification = self.inter_iter_save
		self.replay_memory.add(state=state, q_values=q_values, action=classification, reward=reward)
		self.inter_iter_save = None

	def episode(self, save=True, verbose=False) -> None:
		"""
		at the end of an episode the q-values are updated with the perceived reward
		and the net is trained with the resulting data
		:param save: whether the net is saved after training
		:param verbose: whether additional information will be printed to the console
		:return: None
		"""
		self.replay_memory.update_q_values()
		input_tensors, labels = self.replay_memory.get_training_set()
		self.model.fit(input_tensors, labels,
					batch_size=self.cfg.get_("batch_size"),
					epochs=self.cfg.get_("epochs_per_episode"),
					callbacks=self.tb_callback,
					verbose=int(verbose),
					shuffle=self.cfg.get_("shuffle_data"))

		self.epsilon -= self.cfg.get_("epsilon_decay")
		min_epsilon = self.cfg.get_("min_epsilon")
		if self.epsilon < min_epsilon:
			self.epsilon = min_epsilon

		if save:
			self.save()
		self.write()
		self.replay_memory.write()
		self.replay_memory.clear()

		# time measurement
		self.ep_len_iters.append(self.iteration_count)
		self.iteration_count = 0
		cur_time = time()
		if self.last_episode is not None:
			self.ep_len_secs += cur_time - self.last_episode
		else:
			self.ep_len_secs += sum(self.iter_len_secs)
		self.last_episode = cur_time

	def write(self) -> None:
		"""
		writes information about the current run into the temp directory to allow
		other programs to read it and to save memory
		:return: None
		"""
		with open(self.DEFAULT_TEMP_DIR + "ep_len_iters.csv", "a") as file:
			file.write(str(self.ep_len_iters[-1]) + "\n")
		with open(self.DEFAULT_TEMP_DIR + "ep_len_secs.csv", "a") as file:
			file.write(str(self.ep_len_secs[-1]) + "\n")
		with open(self.DEFAULT_TEMP_DIR + "net_output.csv", "a") as file:
			for entry in self.outputs:
				entry_string = ""
				for val in entry:
					entry_string += str(round(val, 4)) + ", "
				file.write(entry_string.rstrip(", ") + "\n")
			self.outputs = []
		with open(self.DEFAULT_TEMP_DIR + "classifications.csv", "a") as file:
			for entry in self.classifications:
				entry_string = ""
				for val in entry:
					entry_string += str(round(val, 4)) + ", "
				file.write(entry_string.rstrip(", ") + "\n")
			self.classifications = []
		with open(self.DEFAULT_TEMP_DIR + "iter_len_secs.csv", "a") as file:
			for entry in self.iter_len_secs:
				entry_string = ""
				for val in entry:
					entry_string += str(round(val, 4)) + ", "
				file.write(entry_string.rstrip(", ") + "\n")
			self.iter_len_secs = []

	def save(self) -> None:
		"""
		saves all relevant information about the network and the current training run
		1. the current epsilon value is saved to the config as the new starting epsilon
		2. the config is written to the file system
		3. all files from the temp directory are saved to a permanent directory in the saved folder
		:return: None
		"""
		save_dir = self.cfg["save_path"] + self.cfg["name"] + "/"
		self.cfg["epsilon"] = self.epsilon
		self.cfg.save(save_dir)

		self.model.save(save_dir + "/net/")

		for _, _, file in os.walk(self.DEFAULT_TEMP_DIR):
			with open(file, "r") as src, open(save_dir + "stats/" + file, "w") as dest:
				dest.write(src.read())


class QTrainerConfig(ConfigParser):
	def __init__(self, config_file_path: str):
		super().__init__()
		self.read(config_file_path)
		self.type_of = {
			"name": str,
			"save_path": str,
			"epsilon": float,
			"min_epsilon": float,
			"epsilon_decay": float,
			"discount_factor": float,
			"batch_size": int,
			"n_epochs": int,
			"shuffle": to_bool
		}

	def get_(self, key: str) -> object:
		return self.type_of[key](self[key])

	def save(self, save_dir: str):
		self.write(save_dir + "config.cfg")


def to_bool(s: str):
	return s == "True"


class ReplayMemory:
	DEFAULT_TEMP_DIR = "./NeuralNetworks/temp/"

	def __init__(self, discount_factor: float = 0.97, temp_dir: str = DEFAULT_TEMP_DIR):
		self.discount_factor = discount_factor
		self.temp_dir = temp_dir

		self.size = 0
		self.states = []
		self.q_values = []  # each entry is a list of n_actions float values
		self.actions = []
		self.rewards = []

		self.predicted_q_values = []

		self.estimation_errors = []

	def add(self, state, q_values, action, reward):
		"""
		adds a new entry to memory
		:param state: the state of the environment
		:param q_values: the predicted q-values for every possible action in this state
		:param action: the action that was chosen
		:param reward: the reward received for the transition from state to next state through performing action
		:return:
		"""
		self.states.append(state)
		self.q_values.append(q_values)
		self.actions.append(action)
		self.rewards.append(reward)
		self.size += 1

	def update_q_values(self, sarsa=False):
		"""
		calculates the actual q-values from the predicted ones. during play only the predicted values are stored
		as the real ones are not known (they depend on future states). ideally this method is called at the end of
		an episode -> no future states exist/have any influence on the value of current or past states.
		:param sarsa: whether the sarsa (state action reward state action) variant of q-value updates should be used
						the sarsa-variant uses the q-value of the selected next action instead of the highest q-value next action
		:return:
		"""
		if self.size < 1:
			return -1

		# save a copy of the predicted values for analysis
		self.predicted_q_values = self.q_values[:]

		start_time = time()
		self.estimation_errors = np.zeros(shape=[self.size])

		# the q-value of the last step should be the reward in that step
		self.q_values[-1][self.actions[-1]] = self.rewards[-1]

		# the update moves from the present to the past -> from the back to the front of the array
		for i in reversed(range(self.size - 1)):
			action = self.actions[i]
			reward = self.rewards[i]

			# the q-value for the action is composed of the immediate reward + discounted future reward
			if sarsa:
				following_action = self.actions[i + 1]
				action_value = reward + self.discount_factor * self.q_values[i + 1][following_action]
			else:
				action_value = reward + self.discount_factor * np.max(self.q_values[i + 1])
			self.estimation_errors[i] = abs(action_value - self.q_values[i][action])

			# only the q-value for the selected action can be updated
			self.q_values[i][action] = action_value
		end_time = time()

		print("Average estimation error:", Stat(self.estimation_errors))
		return end_time - start_time

	def get_random_batch(self, batch_size, duplicates=False):
		"""
		get a batch of randomly selected state-q_value-pairs
		:param duplicates: whether duplicates are allowed
		:param batch_size: the number of state-q_value-pairs returned; if not enough entries are available
							and not duplicates are allowed less pairs are returned
		:return: a list of states and a list of q-values. corresponding entries have the same index
		"""
		if self.size <= 0:
			return None

		# if the batch size is greater than the size of the memory the entire memory is returned
		if batch_size > self.size - 1 and not duplicates:
			return self.states, self.q_values

		selection = np.random.choice([i for i in range(self.size)], size=batch_size, replace=duplicates)
		states_batch = np.array(self.states)[selection]
		q_values_batch = np.array(self.q_values)[selection]

		return states_batch, q_values_batch

	def get_training_set(self):
		"""
		:return: a set of training data containing the entire contents of this replay memory
		"""
		return self.states, self.q_values

	def clear(self):
		"""
		deletes all values from memory. only the number of actions and the discount factor are conserved
		:return:
		"""
		self.size = 0
		self.states = []
		self.q_values = []
		self.actions = []
		self.rewards = []

		self.estimation_errors = []

	def write(self):
		"""
		saves the contents of this replay memory to the file system
		:return:
		"""
		np.savetxt(self.temp_dir + "estimation_errors.csv", self.estimation_errors, delimiter=",")
		# np.savetxt(TEMP_DIR + "rewards.csv", self.rewards, delimiter=",")		# replaced with lines 187-189
		np.savetxt(self.temp_dir + "q_values.csv", self.q_values, delimiter=",")
		np.savetxt(self.temp_dir + "pred_q_values.csv", self.predicted_q_values, delimiter=",")

		with open(self.temp_dir + "rewards.csv", "a") as rewards_file:
			if len(self.rewards) > 0:
				# avrg_reward = sum(self.rewards) / len(self.rewards)
				rewards_file.write(str(sum(self.rewards)) + "\n")
			else:
				print("No rewards to be written")

		with open(self.temp_dir + "avrg_estimation_errors.csv", "a") as file:
			if len(self.estimation_errors) > 0:
				avrg_est_err = sum(self.estimation_errors) / len(self.estimation_errors)
				file.write(str(avrg_est_err) + "\n")
			else:
				print("No estimation errors to be written")

		states_differentials = []
		prev_state = self.states[0]
		for state in self.states[1:]:
			differential = [round(state[i] - prev_state[i], 4) for i in range(len(state))]
			states_differentials.append(differential)
			prev_state = state
		np.savetxt(self.temp_dir + "state_diffs.csv", states_differentials, delimiter=",")
