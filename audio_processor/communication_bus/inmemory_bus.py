import json
import logging
import threading
import time
from typing import Dict, Any, Callable, List
from datetime import datetime
from queue import Queue, Empty

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InMemoryBus:
    """
    Lightweight in-memory message bus for pub/sub communication.
    Works without any external broker.
    """

    def __init__(self):
        self.callbacks: Dict[str, List[Callable[[str, Dict[str, Any]], None]]] = {}
        self.queue: "Queue[tuple[str, Dict[str, Any]]]" = Queue()
        self.running = False
        self.thread: threading.Thread = None

    def _loop(self):
        """Internal loop to dispatch messages from the queue."""
        while self.running:
            try:
                topic, payload = self.queue.get(timeout=0.5)
                if topic in self.callbacks:
                    for callback in self.callbacks[topic]:
                        try:
                            callback(topic, payload)
                        except Exception as e:
                            logger.error(f"Error in callback for {topic}: {e}")
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in bus loop: {e}")

    def connect(self):
        """Start the bus loop in a background thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            logger.info("InMemoryBus started")

    def disconnect(self):
        """Stop the bus loop."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        logger.info("InMemoryBus stopped")

    def publish(self, topic: str, payload: Dict[str, Any]):
        """Publish a message to a topic."""
        if "timestamp" not in payload:
            payload["timestamp"] = datetime.utcnow().isoformat()
        self.queue.put((topic, payload))
        logger.debug(f"Published to {topic}: {payload}")

    def subscribe(self, topic: str, callback: Callable[[str, Dict[str, Any]], None]):
        """Subscribe a callback to a topic."""
        if topic not in self.callbacks:
            self.callbacks[topic] = []
        if callback not in self.callbacks[topic]:
            self.callbacks[topic].append(callback)
            logger.info(f"Subscribed to {topic}")

    def unsubscribe(self, topic: str, callback: Callable[[str, Dict[str, Any]], None] = None):
        """Unsubscribe from a topic or remove a specific callback."""
        if topic in self.callbacks:
            if callback is None:
                del self.callbacks[topic]
                logger.info(f"Unsubscribed from {topic}")
            else:
                if callback in self.callbacks[topic]:
                    self.callbacks[topic].remove(callback)
                    logger.debug(f"Removed callback for {topic}")
                if not self.callbacks[topic]:
                    del self.callbacks[topic]
                    logger.info(f"Unsubscribed from {topic} (no more callbacks)")


# Example usage
if __name__ == "__main__":
    def on_voice_command(topic: str, payload: Dict[str, Any]):
        print(f"Received voice command on {topic}: {payload}")

    bus = InMemoryBus()
    bus.connect()
    bus.subscribe("voice/commands", on_voice_command)

    example_command = {
        "session_id": "20230902_123456",
        "text": "Hello, this is a test command",
        "confidence": 0.95,
        "metadata": {
            "source": "test_script",
            "language": "en-US"
        }
    }

    bus.publish("voice/commands", example_command)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Disconnecting...")
    finally:
        bus.disconnect()
