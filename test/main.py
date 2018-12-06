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
	screen.blit(player_sprite, (game_time**2 % SCREEN_WIDTH, game_time % SCREEN_HEIGHT))


if __name__ == '__main__':
	print("testing")
	pass
