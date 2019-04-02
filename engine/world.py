import numpy as np
from pygame import Surface
from Util.vector_math import Vector2
from enum import Enum
from engine.tileset import TileSet

DEFAULT_LEVEL_SAVE_PATH = "../resources/levels/"


class Direction(Enum):
	NORTH = 0
	NORTH_EAST = 1
	EAST = 2
	SOUTH_EAST = 3
	SOUTH = 4
	SOUTH_WEST = 5
	WEST = 6
	NORTH_WEST = 7


class World:
	def __init__(self, level_path: str):
		self.tiles = []
		self.tile_set = None
		self.entities = []
		self.tile_entities = []
		self.collision_regions = []
		self.load_level(level_path)

	def tick(self, tick_id: int, tick_time: int):
		for entity in self.entities:
			entity.tick(tick_id, tick_time)
		for tile_entity in self.tile_entities:
			tile_entity.tick(tick_id, tick_time)
		for collision_region in self.collision_regions:
			collision_region.resolve_collisions()

	def render(self, surface: Surface) -> None:
		for x in range(len(self.tiles)):
			for y in range(len(self.tiles[x])):
				surface.blit(self.tiles[x][y].image, (x * 32, y * 32))

	def load_level(self, level_path: str) -> None:
		self.tile_set = TileSet(level_path + "tileset.cfg")
		self.tiles = np.loadtxt(level_path + "tiles.csv", delimiter=",", dtype=str).tolist()
		for x in range(len(self.tiles)):
			for y in range(len(self.tiles[x])):
				self.tiles[x][y] = self.tile_set[self.tiles[x][y]]


class CollisionRegion:
	def __init__(self, top_left: Vector2, bottom_right: Vector2):
		self.top_left = top_left
		self.bottom_right = bottom_right

		self.entities = []
		self.neighbors = {}

	def resolve_collisions(self) -> None:
		pass

	def enter(self):
		pass

	def exit(self):
		pass
