from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import required modules
import asyncio
import logging
from stt_audio_processor.stt_processor import STTAudioProcessorService
from tts_audio_processor.tts_processor import TTSAudioProcessorService
from agents.agent_processor import AgentProcessor
from live_interaction.ui_service import UIService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def async_main():
    """Async main entry point for the Agetic Conversation Bot."""
    
    try:
        logger.info("Starting Agetic Conversation Bot...")
        
        # Initialize services with the shared bus
        # stt_audio_processor = STTAudioProcessorService()
        # tts_audio_processor = TTSAudioProcessorService()
        agent_processor = AgentProcessor()
        ui_service = UIService()

        # Start services
        await asyncio.gather(
            # stt_audio_processor.start(),
            # tts_audio_processor.start(),
            agent_processor.start(),
            ui_service.start()
        )
        
        # Keep the application running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        logger.info("Shutdown signal received...")
    except Exception:
        logger.exception("An error occurred:")
    finally:
        logger.info("Shutting down services...")

        # stop all the services
        await asyncio.gather(
            # stt_audio_processor.stop(),
            # tts_audio_processor.stop(),
            agent_processor.stop(),
            ui_service.stop()
        )
        
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
