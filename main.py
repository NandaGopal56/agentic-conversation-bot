import asyncio
import logging
from audio_processor.main import AudioProcessorService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def async_main():
    """Async main entry point for the Agetic Conversation Bot."""
    audio_processor = None
    try:
        logger.info("Starting Agetic Conversation Bot...")
        
        audio_processor = AudioProcessorService()
        await audio_processor.start()
        
        # Keep the application running
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        logger.info("Shutdown signal received...")
    except Exception as e:
        logger.exception("An error occurred:")
    finally:
        # Cleanup code here
        if audio_processor and hasattr(audio_processor, 'stop'):
            await audio_processor.stop()
        logger.info("Agetic Conversation Bot has been shut down")

def main():
    """Synchronous entry point."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Shutting down Agetic Conversation Bot...")
    except Exception as e:
        logger.exception("An unexpected error occurred:")

if __name__ == "__main__":
    main()
