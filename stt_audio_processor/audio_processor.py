import asyncio
import logging
import os
from typing import Optional
from communication_bus.inmemory_bus import bus, InMemoryBus
from .voice_assistant import VoiceProcessor

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # create the files with dir if not available
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/stt_audio_processor.log")]
)
logger = logging.getLogger(__name__)

class AudioProcessorService:
    """Service to manage the voice assistant and its communication with the message bus."""
    
    def __init__(self):
        """Initialize the audio processor service."""
        self.bus: InMemoryBus = bus
        self.assistant = VoiceProcessor(self.bus)
        self._is_running = False
        self._run_task: Optional[asyncio.Task] = None
    
    async def start(self, **kwargs) -> None:
        """Start the audio processor service asynchronously."""
        if self._is_running:
            logger.warning("Audio processor is already running")
            return
            
        try:
            # Connect to the message bus
            await self.bus.connect()
            
            # Start the assistant in a separate task
            self._is_running = True
            self._run_task = asyncio.create_task(self._run())
            
        except Exception as e:
            logger.error(f"Failed to start audio processor: {e}", exc_info=True)
            self._is_running = False
            await self.bus.disconnect()
            raise
    
    async def _run(self) -> None:
        """Run the main processing loop."""
        try:
            if self.assistant:
                await self.assistant.run()
        except asyncio.CancelledError:
            logger.info("Audio processor run task cancelled")
        except Exception as e:
            logger.error(f"Error in audio processor run task: {e}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop the audio processor service asynchronously."""
        if not self._is_running:
            return
            
        try:
            if self.assistant:
                await self.assistant.stop()
            await self.bus.disconnect()
            
            if self._run_task:
                self._run_task.cancel()
                try:
                    await self._run_task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            logger.error(f"Error stopping audio processor: {e}", exc_info=True)
            raise
        finally:
            self._is_running = False