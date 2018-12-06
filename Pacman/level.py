from enum import Enum
from pygame import Surface
import pygame


class TileType(Enum):
	WALL = 1
	FREE = 0
	DOT = 2


class PacmanLevel:
	def __init__(self, size: tuple =(5, 5)):
		self.tiles = self.init_tiles(size)

		self.wall = pygame.image.load("../resources/square_blue.png")
		self.free = pygame.image.load("../resources/square_black.png")
		self.dot = pygame.image.load("../resources/pacman/dot_32.png")

	@staticmethod
	def init_tiles(size: tuple):
		tiles = []
		for x in range(size[0]):
			column = []
			for y in range(size[1]):
				if x == 0 or y == 0 or x == size[0]-1 or y == size[1]-1:
					column.append(TileType.WALL)
				else:
					column.append(TileType.FREE)
			tiles.append(column)
		return tiles

	def can_walk(self, pos: tuple =(0, 0)) -> bool:
		tile = self.tiles[pos[0]][pos[1]]
		if tile == TileType.FREE or tile == TileType.DOT:
			return True
		return False

	def set(self, x: int, y: int, tile_type: TileType) -> None:
		self.tiles[x][y] = tile_type

	def get_n_dots(self) -> int:
		count = 0
		for tile, _, _ in self.all_tiles():
			if tile == TileType.DOT:
				count += 1
		return count

	def render(self, surface: Surface):
		for x in range(len(self.tiles)):
			for y in range(len(self.tiles[0])):
				tile = self.tiles[x][y]
				if tile == TileType.FREE:
					surface.blit(self.free, x*32, y*32)
				elif tile == TileType.WALL:
					surface.blit(self.wall, x*32, y*32)
				elif tile == TileType.DOT:
					surface.blit(self.dot, x*32, y*32)

	def all_tiles(self):
		for x in range(len(self.tiles)):
			for y in range(len(self.tiles[0])):
				yield self.tiles[x][y], x, y
