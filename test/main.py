import pygame
import os.path as path
from random import randrange
from time import time

SQUARE_GREEN = "../resources/square_green.png"
SCREEN_WIDTH = 512
SCREEN_HEIGHT = 256


def render(screen: pygame.Surface, game_time: int):
	player_sprite = pygame.image.load(SQUARE_GREEN)
	background_color = (40, 40, 40)
	screen.fill(background_color)
	screen.blit(player_sprite, )


def f(t: float = 0):
	x = 24
	# x = t**2 + 5 * t - 2
	y = t**2

	return int(x % SCREEN_WIDTH), int(y % SCREEN_HEIGHT)


def main():
	pygame.init()

	pygame.display.set_caption("minimal program")

	# create a surface on screen that has the size of 240 x 180
	screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

	# main loop
	running = True
	while running:
		t = int(time())
		render(screen, t)
		pygame.display.flip()

		# event handling, gets all event from the eventqueue
		for event in pygame.event.get():
			# only do something if the event is of type QUIT
			if event.type == pygame.QUIT:
				# change the value to False, to exit the main loop
				running = False


if __name__ == '__main__':
	main()
