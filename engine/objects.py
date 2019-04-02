import pygame
from json import dump,loads
from Util.vector_math import Vector2


class Entity:
	def __init__(self, position: Vector2, velocity: Vector2 =Vector2(0, 0)):
		self.pos = position
		self.vel = velocity
		self.acc = Vector2(0, 0)
		self.friction = Vector2(0, 0)

		self.img = pygame.image.load("../resources/square_green.png")

	def tick(self, tick_id: int, tick_time: int) -> None:
		self.vel = self.acc.scalar_mul(abs(self.acc - self.friction))
		self.pos += self.vel

	def render(self, surface: pygame.Surface) -> None:
		surface.blit(self.img, self.pos.as_tuple())


class Tile:
	def __init__(self, tile_id: int, image_path: str, collision: bool = False):
		self.id = tile_id
		self.image = pygame.image.load(image_path)
		self.collision = collision

	@staticmethod
	def from_json(tile_rep):
		d = loads(tile_rep)
		return Tile(d["id"], d["image_path"], d["collision"])
