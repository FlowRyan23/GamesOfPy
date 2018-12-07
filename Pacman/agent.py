import pygame
from enum import Enum
from Pacman.constants import PROJECT_ROOT


class AgentType(Enum):
	PACMAN = 1
	GHOST = 2


class PacmanAction(Enum):
	QUIT = -1
	PASS = 0
	MOVE_UP = 1
	MOVE_RIGHT = 2
	MOVE_DOWN = 3
	MOVE_LEFT = 4


class Agent:
	def __init__(self):
		self.pos_x = 1
		self.pos_y = 1

		self.type = None
		self.no_texture = pygame.image.load(PROJECT_ROOT + "/resources/square_green.png")
		self.pacman = pygame.image.load(PROJECT_ROOT + "/resources/pacman/pacman_32.png")
		self.ghost = pygame.image.load(PROJECT_ROOT + "/resources/square_green.png")

	def render(self, surface: pygame.Surface) -> None:
		if self.type == AgentType.PACMAN:
			surface.blit(self.pacman, (self.pos_x*32, self.pos_y*32))
		elif self.type == AgentType.GHOST:
			surface.blit(self.ghost, (self.pos_x*32, self.pos_y*32))
		else:
			surface.blit(self.no_texture, (self.pos_x*32, self.pos_y*32))

	def act(self, game_state) ->PacmanAction:
		return PacmanAction.QUIT
