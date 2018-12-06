from Pacman.agent import Agent, AgentType, PacmanAction
from random import choice


class RandomGost(Agent):
	def __init__(self):
		super().__init__()
		self.type = AgentType.GHOST
		self.choices = [PacmanAction.MOVE_UP,
						PacmanAction.MOVE_DOWN,
						PacmanAction.MOVE_LEFT,
						PacmanAction.MOVE_RIGHT]

	def act(self):
		return choice(self.choices)
