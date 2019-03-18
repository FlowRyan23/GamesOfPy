from pygame.event import Event


class InputHandler:
	def __init__(self):
		pass

	def handle(self, event: Event) -> None:
		print(event)