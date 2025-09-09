import asyncio
import logging
from audio_processor.main import AudioProcessorService
from communication_bus.inmemory_bus import InMemoryBus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def async_main():
    """Async main entry point for the Agetic Conversation Bot."""
    audio_processor = None
    stop_event = asyncio.Event()

    try:
        logger.info("Starting Agetic Conversation Bot...")

        audio_processor = AudioProcessorService()

        # Start services
        await audio_processor.start()

        # Wait until stop_event is set (instead of while True loop)
        await stop_event.wait()

    except asyncio.CancelledError:
        logger.info("Shutdown signal received...")
    except Exception:
        logger.exception("An error occurred:")
    finally:
        if audio_processor and hasattr(audio_processor, 'stop'):
            await audio_processor.stop()
        logger.info("Agetic Conversation Bot has been shut down")


def main():
    """Synchronous entry point."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Shutting down Agetic Conversation Bot...")
    except Exception:
        logger.exception("An unexpected error occurred:")


if __name__ == "__main__":
    main()
