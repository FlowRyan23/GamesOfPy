import pygame
from Pacman.constants import RESOURCE_FOLDER, SCREEN_HEIGHT, SCREEN_WIDTH
from Pacman.agent import PacmanAction, AgentType
from Pacman.level import PacmanLevel, TileType
from Pacman.agents.ghosts import RandomGost
from Pacman.agents.Neuroman import Neuroman


class PacmanGame:
	def __init__(self):
		self.level = PacmanLevel(level_data_path=RESOURCE_FOLDER + "pacman/levels/large_a.txt")

		self.agents = []
		self.agents.append(RandomGost())
		pacman = Neuroman(self.get_game_state_size())
		self.agents.append(pacman)

		self.move_offsets = {
			PacmanAction.MOVE_UP: (0, -1),
			PacmanAction.MOVE_DOWN: (0, 1),
			PacmanAction.MOVE_LEFT: (-1, 0),
			PacmanAction.MOVE_RIGHT: (1, 0)
		}

		pygame.init()
		pygame.display.set_caption("Pacman")
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

	def run(self):
		"""
		main game loop. runs until the game is over (see self.game_over()) or the window is closed
		:return:
		"""
		running = True
		while not self.game_over() and running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False

			for agent in self.agents:
				action = agent.act(game_state=self)
				if action == PacmanAction.QUIT:
					self.agents.remove(agent)
				elif action == PacmanAction.PASS:
					continue
				else:
					x_offset, y_offset = self.move_offsets[action]
					x, y = agent.pos_x + x_offset, agent.pos_y + y_offset

					if self.level.can_walk((x, y)):
						agent.pos_x, agent.pos_y = x, y
						if agent.type == AgentType.PACMAN:
							self.level.set(x, y, TileType.FREE)

				for other in self.agents:
					same_pos = agent.pos_x == other.pos_x and agent.pos_y == other.pos_y
					killable = agent.type == AgentType.PACMAN and other.type == AgentType.GHOST
					if other is not agent and same_pos and killable:
						self.agents.remove(agent)

			self.render()
		print("Game Over")

	def render(self) -> None:
		"""
		draws the level and all agents on screen
		:return:
		"""
		self.level.render(self.screen)
		for agent in self.agents:
			agent.render(self.screen)
		pygame.display.flip()

	def game_over(self) -> bool:
		"""
		test whether the game is over.
		the game is over if at least one is true:
			1. all agents have died			-> lost
			2. all dots have been eaten		-> won
		:return: true if the game is over
		"""
		no_more_dots = self.level.get_n_dots() <= 0
		no_more_agents = len(self.agents) <= 0
		if no_more_agents or no_more_dots:
			return True
		return False

	def get_game_state(self):
		"""
		creates a list of float values between -1.0 and 1.0 that represents the curent
		state of the game fully.
		All values are normalized to be fed into a neural net
		:return: the state of the game
		"""
		state = []
		for tile, _, _ in self.level.all_tiles():
			state.append(tile.value / 2.0)
		for agent in self.agents:
			state.append(agent.type.value / 2.0)
			state.append(round(agent.pos_x / float(self.level.size[0]), 2))
			state.append(round(agent.pos_y / float(self.level.size[1]), 2))
		return state

	def get_game_state_size(self):
		return self.level.size[0] * self.level.size[1] + len(self.agents) * 3


if __name__ == '__main__':
	PacmanGame().run()
