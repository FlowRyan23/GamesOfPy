import pygame
from time import time
from enum import Enum
from engine.world import World
from engine.input_handling import InputHandler

DEFAULT_SCREEN_WIDTH = 720
DEFAULT_SCREEN_HEIGHT = 1080


class GameOverMode(Enum):
	QUIT = -1		# closes the window and terminates the program
	STOP = 0		# stops the game, but leaves the window open
	PAUSE = 1		# the game is paused and can be continued afterwards
	RESET = 2		# the game is reset into its starting state
	MENU = 3		# the main/parent menu is opened


class GameLoop:
	def __init__(self, title: str = "game", width: int = DEFAULT_SCREEN_WIDTH, height: int = DEFAULT_SCREEN_HEIGHT) -> None:
		self.running = False
		self.paused = False
		self.tick_count = 0

		pygame.init()
		pygame.display.set_caption(title)
		self.screen = pygame.display.set_mode((width, height))

		self.input_handler = None
		self.world = None
		self.game_over_mode = GameOverMode.QUIT
		self.game_over_ops = {
			GameOverMode.QUIT: self.quit,
			GameOverMode.STOP: self.pause,		# todo make not unpausable
			GameOverMode.PAUSE: self.pause,
			GameOverMode.RESET: self.reset,
			GameOverMode.MENU: self.pause		# todo implement menus
		}

	def run(self, world: World, input_handler: InputHandler, game_over_mode: GameOverMode = GameOverMode.QUIT) -> None:
		"""
		main game loop. runs until the game is over (see self.is_game_over()) or the window is closed
		:return:
		"""
		self.world = world
		self.input_handler = input_handler
		self.game_over_mode = game_over_mode

		self.running = True
		while self.running:
			if not self.paused:
				# todo fix the tickrate to be independent of framerate
				self.tick()
			self.render()

			if self.is_game_over():
				self.game_over_ops[self.game_over_mode]()

		print("Done")

	def tick(self) -> None:
		self.tick_count += 1
		tick_time = time()

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				self.running = False
				# todo calls to release all resources
				return
			elif 0 < event.type < 8:
				self.input_handler.handle(event)

		self.world.tick(self.tick_count, tick_time)

	def render(self) -> None:
		"""
		draws the level and all agents on screen
		:return:
		"""
		self.world.render(self.screen)
		pygame.display.flip()

	def reset(self):
		self.world.reset()

	def is_game_over(self) -> bool:
		"""
		test whether the game is over.
		:return: true if the game is over
		"""
		# todo implement
		return False

	def quit(self):
		self.running = False

	def pause(self):
		self.paused = True


if __name__ == '__main__':
	game = GameLoop()
	game.run(world=World("../resources/levels/empty_square_25/"), input_handler=InputHandler())
