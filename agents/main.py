
from communication_bus.inmemory_bus import InMemoryBus
from .receiver import on_voice_command

bus = InMemoryBus()
bus.connect()
bus.subscribe("voice/commands", on_voice_command)