import pygame
import os
from Pacman.level import PacmanLevel, TileType
from Pacman.agents.ghosts import RandomGost
from Pacman.agent import PacmanAction, AgentType

SCREEN_WIDTH = 512
SCREEN_HEIGHT = 512
PROJECT_ROOT = os.path.abspath("../")
GAME_FOLDER = os.path.abspath("./")


class PacmanGame:
	def __init__(self):
		self.level = PacmanLevel()

		self.agents = []
		self.agents.append(RandomGost())

		self.move_offsets = {
			PacmanAction.MOVE_UP: (0, -1),
			PacmanAction.MOVE_DOWN: (0, 1),
			PacmanAction.MOVE_LEFT: (-1, ),
			PacmanAction.MOVE_RIGHT: (1, 0)
		}

		pygame.init()
		pygame.display.set_caption("Pacman")

		# create a surface on screen that has the size of 240 x 180
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

	def run(self):
		running = True
		while not self.game_over() and running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False

			for agent in self.agents:
				action = agent.act()
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

	def render(self) -> None:
		self.level.render(self.screen)
		for agent in self.agents:
			agent.render(self.screen)
		pygame.display.flip()

	def game_over(self) -> bool:
		no_more_dots = self.level.get_n_dots() <= 0
		no_more_agents = len(self.agents)
		if no_more_agents or no_more_dots:
			return True
		return False


if __name__ == '__main__':
	PacmanGame().run()
