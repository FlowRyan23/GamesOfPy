from enum import Enum
from pygame import Surface
import pygame
import numpy as np
from Util.simple_functions import in_bounds_nd as in_bounds


class TileType(Enum):
	WALL = 1
	FREE = 0
	DOT = 2

	@staticmethod
	def get(tile_type: str):
		if tile_type == "W":
			return TileType.WALL
		elif tile_type == "F" or tile_type == "E":
			return TileType.FREE
		else:
			return TileType.DOT


class PacmanLevel:
	def __init__(self, size: tuple =(5, 5), level_data_path: str = None):
		self.level_data_path = level_data_path
		if level_data_path is None:
			self.tiles = self.init_tiles(size)
		else:
			self.load(level_data_path)
		self.size = (len(self.tiles), len(self.tiles[0]))
		self.n_total_dots = self.get_n_dots()

		print("Level: width={0:d}, height={1:d}".format(len(self.tiles), len(self.tiles[0])))

		self.wall = pygame.image.load("../resources/square_blue.png")
		self.free = pygame.image.load("../resources/square_black.png")
		self.dot = pygame.image.load("../resources/pacman/dot_32.png")

	@staticmethod
	def init_tiles(size: tuple) -> list:
		"""
		creates a level filled with dots and with walls around the edges
		:param size: size of the level
		:return: a new level
		"""
		tiles = []
		for x in range(size[0]):
			column = []
			for y in range(size[1]):
				if x == 0 or y == 0 or x == size[0]-1 or y == size[1]-1:
					column.append(TileType.WALL)
				else:
					column.append(TileType.DOT)
			tiles.append(column)
		return tiles

	def can_walk(self, pos: tuple =(0, 0)) -> bool:
		"""
		checks if the given position is walkable (is of type free or dot)
		:param pos: position to be checked (x, y)
		:return: true if the tile is walkable
		"""
		if not in_bounds(pos, (0, 0), self.size):
			return False

		tile = self.tiles[pos[0]][pos[1]]
		if tile == TileType.FREE or tile == TileType.DOT:
			return True
		return False

	def set(self, x: int, y: int, tile_type: TileType) -> None:
		"""
		sets the tile at the given position to the given tile type
		:param x: x coordinate of the changed tile
		:param y: y coordinate of the changed tile
		:param tile_type: new type for the tile at (x, y)
		"""
		self.tiles[x][y] = tile_type

	def get_n_dots(self) -> int:
		"""
		counts all remaining dots
		:return: number of dots in this level
		"""
		count = 0
		for tile, _, _ in self.all_tiles():
			if tile == TileType.DOT:
				count += 1
		return count

	def get_n_walkable(self) -> int:
		"""
		counts the number of walkable tiles in this level
		free tiles and tiles with dots are walkable
		:return:
		"""
		count = 0
		for tile, _, _ in self.all_tiles():
			if tile == TileType.FREE or tile == TileType.DOT:
				count += 1
		return count

	def render(self, surface: Surface) -> None:
		"""
		renders the level with 32x32 pixels per tile
		:param surface: a pygame surface that can render images
		:return:
		"""
		for tile, x, y in self.all_tiles():
			if tile == TileType.FREE:
				surface.blit(self.free, (x*32, y*32))
			elif tile == TileType.WALL:
				surface.blit(self.wall, (x*32, y*32))
			elif tile == TileType.DOT:
				surface.blit(self.dot, (x*32, y*32))

	def all_tiles(self, as_values: bool =False):
		"""
		generator used to iterate over all tiles
		:param as_values: if true the TileType will be given as integer
		:return: the tile and its x and y coordinates
		"""
		for x in range(len(self.tiles)):
			for y in range(len(self.tiles[0])):
				if as_values:
					yield self.tiles[x][y].value, x, y
				else:
					yield self.tiles[x][y], x, y

	def as_values(self) -> np.ndarray:
		"""
		creates an array with the values of TileType instead of enum
		:return: integer array of TileType.value representing the level
		"""
		lvl = np.zeros(self.size)
		for tile, x, y in self.all_tiles(as_values=True):
			lvl[x][y] = tile
		return lvl

	def load(self, level_path: str, verbose=False) -> None:
		"""
		loads a level from the file system.
		levels are saved as a space separated list of characters (W for wall, F for free and D for dot)
		the first line of the file describes the size of the level as width and height
		:param verbose: en-/disables additional information via print()
		:param level_path: path to the file with the level data
		"""
		self.tiles = []
		with open(level_path, "r") as level_data:
			height = int(level_data.readline().split(" ")[1])

			for y in range(height):
				row_data = level_data.readline().replace(" \n", "").split(" ")
				if verbose:
					print("row:", row_data)
				row = []
				for tile in row_data:
					row.append(TileType.get(tile))
				self.tiles.append(row)

			self.tiles = np.transpose(self.tiles)

	def reset(self) -> None:
		"""
		restores the state of the level at the beginning of the game by loading
		the level data from file system and overriding current
		:return:
		"""
		self.load(self.level_data_path)

	def __str__(self) -> str:
		return str(self.as_values())
