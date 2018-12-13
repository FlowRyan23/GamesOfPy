import numpy as np
from time import time
from Pacman.constants import GAME_FOLDER

TEMP_DIR = GAME_FOLDER + "agents/temp/"
LOG_DIR = GAME_FOLDER + "agents/logs/"


class RunInfo:
	def __init__(self):
		self.run_start_time = time()
		self.total_time = 0

		self.iteration_count = 0
		self.last_iter_time = time()
		self.action_stat = {}
		self.net_output = []

		self.episode_count = 0
		self.last_ep_time = time()
		self.last_ep_iter = 0
		self.episode_lengths = []
		self.episode_times = []
		self.mem_up_times = []
		self.train_times = []

		self.reward_data = {"re_dots": []}

	def episode(self, mem_up_time, train_time, verbose=True):
		ep_time = time()-self.last_ep_time
		self.episode_times.append(ep_time)
		self.total_time += ep_time
		self.episode_lengths.append(self.iteration_count-self.last_ep_iter)
		self.mem_up_times.append(mem_up_time)
		self.train_times.append(train_time)
		self.episode_count += 1

		if verbose:
			print("\nEpisode {0:d} over".format(self.episode_count - 1))
			print("Total episode time:", self.episode_times[-1])
			print("Iterations this episode:", self.episode_lengths[-1])
			print("Memory update took {0:.2f}sec".format(mem_up_time))
			print("training took {0:.2f}sec".format(train_time))
			print("Action stat:", self.action_stat)

		self.last_ep_iter = self.iteration_count
		self.last_ep_time = time()

	def iteration(self, net_output, action, verbose=False):
		self.net_output.append(net_output)

		try:
			self.action_stat[action] += 1
		except KeyError:
			self.action_stat[action] = 1

		self.iteration_count += 1

		if verbose:
			print("Iteration:", self.iteration_count)
			print("Action:", action)

	def reward(self, dots):
		self.reward_data["re_dots"].append(dots)

	def write(self):
		with open(TEMP_DIR + "episode_lengths.csv", "a") as file:
			file.write(str(self.episode_lengths[-1]) + "\n")
		with open(TEMP_DIR + "episode_times.csv", "a") as file:
			file.write(str(self.episode_times[-1]) + "\n")
		with open(TEMP_DIR + "mem_up_times.csv", "a") as file:
			file.write(str(self.mem_up_times[-1]) + "\n")
		with open(TEMP_DIR + "train_times.csv", "a") as file:
			file.write(str(self.train_times[-1]) + "\n")
		with open(TEMP_DIR + "net_output.csv", "a") as file:
			for entry in self.net_output:
				entry_string = ""
				for val in entry:
					entry_string += str(round(val, 4)) + ", "
				file.write(entry_string.rstrip(", ") + "\n")
			self.net_output = []
		with open(TEMP_DIR + "reward_info.csv", "a") as file:
			n_entries = len(self.reward_data["re_dots"])
			for i in range(n_entries):
				entry_string = ""
				for key in self.reward_data.keys():
					entry_string += str(round(self.reward_data[key][i], 4)) + ", "
				file.write(entry_string.rstrip(", ") + "\n")
			self.reward_data = {"re_dots": []}
		# todo action_stat

	def restore(self, bot_id):
		bot_dir = LOG_DIR + bot_id + "/"

		self.episode_lengths = np.loadtxt(bot_dir + "episode_lengths.csv", delimiter=",").tolist()
		self.episode_times = np.loadtxt(bot_dir + "episode_times.csv", delimiter=",").tolist()
		self.mem_up_times = np.loadtxt(bot_dir + "mem_up_times.csv", delimiter=",").tolist()
		self.train_times = np.loadtxt(bot_dir + "train_times.csv", delimiter=",").tolist()
		reward_info = np.loadtxt(bot_dir + "reward_info.csv", delimiter=",").transpose().tolist()
		for i, key in enumerate(self.reward_data.keys()):
			self.reward_data[key] = reward_info[i]
		self.net_output = np.loadtxt(bot_dir + "net_output.csv", delimiter=",").tolist()
		# todo action_stat
		# todo self.state_score_data

		self.total_time = sum(self.episode_lengths)
		self.episode_count = len(self.episode_lengths)
		# todo last ep iter = iteration
		# todo action stat
