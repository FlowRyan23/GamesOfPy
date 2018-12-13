import os
import random
import numpy as np
import pygame
from time import time
from pygame import Surface
from configparser import ConfigParser
from Pacman.constants import PROJECT_ROOT, GAME_FOLDER

from Pacman.agent import Agent, AgentType, PacmanAction
import NeuralNetworks.q_learning as nets
from NeuralNetworks.information import RunInfo
from NeuralNetworks.q_learning import NeuralNetwork, ReplayMemory

# load/save behavior
TRAIN = True
SAVE_NET = True
COLLECT_DATA = True
SAVE_DATA = COLLECT_DATA and True
LOAD = False
PRESERVE = True
LOAD_BOT_NAME = "no name"
NET_PATH = PROJECT_ROOT + "NeuralNetworks/saved/"
TEMP_DIR = GAME_FOLDER + "agents/temp/"
LOG_DIR = GAME_FOLDER + "agents/logs/"

# info
INFO = True
EPISODE_INFO = False
SPAM_INFO = False
INFO_INTERVAL = 10.0					# in seconds
FULL_SAVE_INTERVAL = 5					# how often the bot along with all collected data is saved (in iterations)
SAVE_INTERVAL = 2						# how often the bot is saved (in iterations)

# net and training properties
NET_NAME = "Neuroman" + str(int(time()))
N_OUTPUT = 4
START_EPSILON = 0.9						# chance that a random action will be chosen instead of the one with highest q_value
EPSILON_DECAY = 2e-3					# amount the epsilon value decreases every episode (default 5e-4)
EPSILON_STARTUP_DECAY = 0				# amount the epsilon value decreases every time the bot is loaded (not the first time)
MIN_EPSILON = 0.1						# minimum epsilon value
USE_SARSA = True
RELATIVE_COORDINATES = True
ALLOW_NEGATIVE_REWARD = True

# rewards
RE_DOTS = False
RE_DOTS_EATEN = False
RE_GAME_OVER = True
REWARDS = [RE_DOTS, RE_DOTS_EATEN, RE_GAME_OVER]
REWARD_EXP = 1


class Neuroman(Agent):
	def __init__(self, view_range=5):
		super().__init__()
		self.type = AgentType.PACMAN

		if not TRAIN:
			print("\n-----NOT TRAINING-----\n")

		# clear the contents of temp info files and the log
		clear_temp()
		with open(LOG_DIR + "log.txt", "w") as f:
			f.write("")

		self.name = NET_NAME
		self.prev_info_time = time()  # used to keep track of time since last info
		self.actions = [PacmanAction.MOVE_UP, PacmanAction.MOVE_DOWN, PacmanAction.MOVE_LEFT, PacmanAction.MOVE_RIGHT]
		self.view_range = view_range

		self.epsilon = START_EPSILON
		self.epsilon_decay = EPSILON_DECAY
		self.sarsa = USE_SARSA
		self.rel_coords = RELATIVE_COORDINATES
		self.reward_exp = REWARD_EXP
		self.neg_reward = ALLOW_NEGATIVE_REWARD

		self.aps = 0  # actions per second
		self.run_info = RunInfo()

		# list of tuples of (state, action, reward);
		self.replay_memory = ReplayMemory(n_actions=N_OUTPUT)
		self.prev_state = None
		self.prev_input_tensor = None
		self.prev_action = None
		self.prev_q_values = None
		self.reward_accumulator = 0  # for reward functions that need more than one iteration

		self.user_input_cool_down = time()

		if LOAD:
			self.load(preserve=PRESERVE)
		else:
			drop_out_rate = 0.2
			self.net = nets.flat_3(NET_NAME, in_shape=[(2*view_range + 1) * (2*view_range + 1)], n_classes=len(self.actions), drop_out=drop_out_rate)

		# the net is ready to be called
		self.net_ready = True
		print(self.name, "ready\n")

	def act(self, game_state):
		self.aps += 1

		# set the reward for the previous iteration; not possible in the first iteration because not previous state and action are available
		if self.prev_state is not None and self.prev_q_values is not None and self.prev_action is not None and self.prev_input_tensor is not None:
			reward = self.online_reward(game_state)
			self.replay_memory.add(state=self.prev_input_tensor,
								   q_values=self.prev_q_values,
								   action=self.prev_action.value - 1,
								   reward=reward)

		input_tensor = self.form_state(game_state).flatten()
		# print("tensor shape", np.shape(input_tensor))
		predicted_q_values = self.net.run([input_tensor])
		if random.random() < self.epsilon:
			chosen_class = random.randrange(0, len(predicted_q_values))
		else:
			chosen_class = np.argmax(predicted_q_values)
		action = self.actions[chosen_class]

		self.prev_state = game_state
		self.prev_input_tensor = input_tensor
		self.prev_q_values = predicted_q_values
		self.prev_action = action

		# info for debugging purposes
		cur_time = time()
		if INFO and cur_time - self.prev_info_time > INFO_INTERVAL:
			print("\n------------------------Info------------------------")
			print("Running", self.aps / INFO_INTERVAL, "a/s")
			print("Episode", self.run_info.episode_count)
			# print("Game state:", game_info)
			print("Epsilon:", round(self.epsilon, 5))
			# print("Memory size:", self.replay_memory.size)
			# print("Net input:", state)
			# print("Net Output:", str(predicted_q_values))
			# print("Action:", selected_action)
			# print("Return Vector:", return_controller_state)
			# print("------------------------------------------------------")
			# print()

			self.prev_info_time = cur_time  # set time of previous info to now
			self.aps = 0  # reset actions per second

		self.run_info.iteration(net_output=predicted_q_values, action=action.value-1, verbose=SPAM_INFO)

		return action

	def next_episode(self):
		"""
		updates the q_values in the replay memory and trains the net
		:return: the time it took to 1: update the qvs and 2: train the net
		"""
		# decrease epsilon
		self.epsilon = self.epsilon - self.epsilon_decay
		if self.epsilon < MIN_EPSILON:
			self.epsilon = MIN_EPSILON

		mem_up_time, train_start, train_end = 0, 0, 0
		if TRAIN:
			mem_up_time = self.replay_memory.update_q_values(sarsa=self.sarsa)

			train_start = time()
			self.net.train(self.replay_memory.get_training_set(), batch_size=512, n_epochs=4, save=SAVE_NET)
			train_end = time()

		self.run_info.episode(mem_up_time, train_end - train_start, verbose=EPISODE_INFO)
		# write data to file
		if COLLECT_DATA:
			self.replay_memory.write()
			self.run_info.write()

		# todo research different strategies for replay memory
		self.replay_memory.clear()

		if self.run_info.episode_count % FULL_SAVE_INTERVAL == 0:
			self.save(info_files=True)
		elif self.run_info.episode_count % SAVE_INTERVAL == 0:
			self.save()

	def online_reward(self, cur_game_state):
		"""
		calculates the reward the agent receives for transitioning from one state(self.prev_game_info) to another(cur_game_info) using the chosen action
		:param cur_game_state: the state the agent moved to
		:return: the reward for the change in state
		"""
		reward = 0

		if RE_DOTS:
			n_dots = cur_game_state.level.get_n_dots()
			n_tiles = cur_game_state.level.get_n_walkable()

			reward += 1 - n_dots/n_tiles

		return reward**REWARD_EXP

	def end_reward(self):
		reward = 0
		if RE_DOTS_EATEN:
			de_comp = 1 - self.prev_state.level.get_n_dots()/self.prev_state.level.n_total_dots
			reward += de_comp

		if RE_GAME_OVER:
			if self.alive:
				reward += 1
			else:
				reward -= 1
		return reward**REWARD_EXP

	def form_state(self, game_state):
		state = np.zeros([2*self.view_range + 1, 2*self.view_range + 1])
		x_offset, y_offset = self.view_range - self.pos_x, self.view_range - self.pos_y

		level = np.array(game_state.level.as_values()) / 2
		width, height = game_state.level.size
		state[x_offset: x_offset+width, y_offset: y_offset+height] = level

		for agent in game_state.agents:
			if agent is not self and agent.alive:
				state[agent.pos_x + x_offset][agent.pos_y + y_offset] = -1 if agent.type == AgentType.GHOST else 0

		# print("level")
		# print(game_state.level)
		# print("state")
		# for row in state:
		# 	print(row)
		return state

	def net_view(self, surface: Surface) -> None:
		state = self.form_state(self.prev_state)
		free, wall, dot, pacman, ghost = self.prev_state.level.free, self.prev_state.level.wall, self.prev_state.level.dot, self.pacman, self.no_texture
		for x in range(len(state)):
			for y in range(len(state[0])):
				tile = state[x][y]
				if tile == -1:
					surface.blit(ghost, (x * 32, y * 32))
				elif tile == 0:
					surface.blit(free, (x * 32, y * 32))
				elif tile == 0.5:
					surface.blit(wall, (x * 32, y * 32))
				elif tile == 1:
					surface.blit(dot, (x * 32, y * 32))

		x_offset, y_offset = self.view_range - self.pos_x, self.view_range - self.pos_y
		surface.blit(pacman, ((self.pos_x+x_offset) * 32, (self.pos_y+y_offset) * 32))
		pygame.display.flip()

	def reset(self):
		reward = self.end_reward()
		print("end reward:", reward)
		self.replay_memory.add(state=self.prev_input_tensor,
							   q_values=self.prev_q_values,
							   action=self.prev_action.value - 1,
							   reward=reward)
		self.next_episode()
		self._reset()

	def retire(self):
		print("retire")
		if SAVE_NET:
			self.net.save()
		self.net.close()
		if SAVE_DATA:
			self.save(info_files=True)

	def save(self, info_files=False):
		run_indexer = ConfigParser()
		run_indexer.read(LOG_DIR + "run_index.cfg")

		try:
			run_indexer[self.name]["reward"] = str(REWARDS).replace("[", "").replace("]", "")
			run_indexer[self.name]["reward_exp"] = str(self.reward_exp)
			run_indexer[self.name]["neg_reward"] = str(self.neg_reward)
			run_indexer[self.name]["n_episodes"] = str(self.run_info.episode_count)
			run_indexer[self.name]["epsilon"] = str(self.epsilon)
			run_indexer[self.name]["epsilon_decay"] = str(self.epsilon_decay)
			run_indexer[self.name]["sarsa"] = str(self.sarsa)
			run_indexer[self.name]["relative_coordinates"] = str(self.rel_coords)
			run_indexer[self.name]["description"] = "- auto generated description -"
		except KeyError:
			run_indexer[self.name] = {
				"reward": str(REWARDS).replace("[", "").replace("]", ""),
				"reward_exp": str(self.reward_exp),
				"neg_reward": str(self.neg_reward),
				"n_episodes": str(self.run_info.episode_count),
				"epsilon": str(self.epsilon),
				"epsilon_decay": str(self.epsilon_decay),
				"sarsa": str(self.sarsa),
				"relative_coordinates": str(self.rel_coords),
				"description": "- auto generated description -"
			}

		with open(LOG_DIR + "run_index.cfg", "w") as ri_file:
			run_indexer.write(ri_file)

		if not os.path.isdir(LOG_DIR + self.name):
			os.makedirs(LOG_DIR + self.name)

		if info_files:
			# copy the temp-files into logs folder
			for _, _, files in os.walk(TEMP_DIR):
				for file in files:
					with open(TEMP_DIR + file, "r") as src, open(LOG_DIR + self.name + "/" + file, "w") as dest:
						dest.write(src.read())

	def load(self, bot_name=LOAD_BOT_NAME, preserve=True):
		# read bot information from file
		run_indexer = ConfigParser()
		run_indexer.read(LOG_DIR + "run_index.cfg")
		net_info = run_indexer[bot_name]

		# determine the name for the bot
		if preserve:
			new_name = name_increment(bot_name)
			print("finding name")
			while os.path.isdir(LOG_DIR + new_name):
				print(new_name, "was not available")
				new_name = name_increment(new_name)
			print("name found")
		else:
			new_name = bot_name

		# reset attributes which may be set incorrectly in constructor
		self.name = new_name
		self.epsilon = max(0.1, float(net_info["epsilon"]) - EPSILON_STARTUP_DECAY)
		self.epsilon_decay = float(net_info["epsilon_decay"])
		self.replay_memory = ReplayMemory(n_actions=len(self.actions))
		self.sarsa = net_info["sarsa"] == "True"
		self.reward_exp = int(net_info["reward_exp"])
		self.neg_reward = net_info["neg_reward"] == "True"
		self.rel_coords = net_info["relative_coordinates"] == "True"

		self.net = NeuralNetwork.restore(bot_name, new_name=new_name, verbose=True)
		self.run_info.restore(bot_name)

		# copy information files from log into active (temp) folder
		bot_dir = LOG_DIR + bot_name + "/"
		for _, _, files in os.walk(bot_dir):
			for file in files:
				with open(bot_dir + file, "r") as src, open(TEMP_DIR + file, "w") as dest:
					dest.write(src.read())

	def __str__(self):
		return self.name


def name_increment(net_name):
	"""
	produces a new net name by incrementing its sub id character
	e.g.:
		- FlowBot15313548 becomes FlowBot15313548/a
		- FlowBot15313548/a becomes FlowBot15313548/b
		- etc.
	:param net_name: the old name
	:return: the new name
	"""
	new_name = net_name.split("_")
	if len(new_name) == 1:
		new_name = new_name[0] + "_a"
	elif len(new_name) == 2:
		new_name = new_name[0] + "_" + chr(ord(new_name[1]) + 1)
	else:
		raise ValueError("invalid net name")
	return new_name


def clear_temp():
	for _, _, files in os.walk(TEMP_DIR):
		for file in files:
			with open(TEMP_DIR + file, "w") as tmp:
				tmp.write("")
