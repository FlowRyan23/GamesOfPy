from configparser import ConfigParser
from engine.objects import Tile

DEFAULT_TILESET = {
	"0": Tile(0, "../resources/square_black.png", collision=True),
	"1": Tile(1, "../resources/square_blue.png", collision=True)
}

BUILT_IN_TILESETS = {
	"default": DEFAULT_TILESET
}


# todo load on request, caching
class TileSet:
	def __init__(self, file_path):
		self.resources = ConfigParser()
		self.resources.read(file_path)
		self.tiles = {}

		try:
			print(self.resources["DEFAULT"]["tileset_id"])
			self.tiles = BUILT_IN_TILESETS[self.resources["DEFAULT"]["tileset_id"]]
		except KeyError:
			print("no build in tileset found")
			self.tiles = self.tiles["TILES"]
			for tile_id in self.resources.keys():
				self.tiles[tile_id] = Tile.from_json(self.resources[tile_id])

	def __getitem__(self, item):
		return self.tiles[item]
