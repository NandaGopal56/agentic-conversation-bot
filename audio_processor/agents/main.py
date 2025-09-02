from communication_bus.inmemory_bus import InMemoryBus


def on_voice_command(topic: str, payload: Dict[str, Any]):
    print(f"Received voice command on {topic}: {payload}")

bus = InMemoryBus()
bus.connect()
bus.subscribe("voice/commands", on_voice_command)
