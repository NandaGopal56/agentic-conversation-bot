import argparse
import asyncio
import logging
import sys
import traceback
from sys import platform
from typing import Optional

from communication_bus.inmemory_bus import InMemoryBus
from .config import AUDIO_CONFIG, MODEL_CONFIG, WAKE_WORD_CONFIG, SYSTEM_CONFIG
from .voice_assistant import VoiceProcessor

logger = logging.getLogger(__name__)

class AudioProcessorService:
    """Service to manage the voice assistant and its communication with the message bus."""
    
    def __init__(self):
        """Initialize the audio processor service."""
        self.bus = InMemoryBus()
        self.assistant: Optional[VoiceProcessor] = None
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
            
            # Initialize voice assistant with message bus
            self.assistant = VoiceProcessor()
            
            # Start the assistant in a separate task
            self._is_running = True
            self._run_task = asyncio.create_task(self._run())
            
        except Exception as e:
            logger.error(f"Failed to start audio processor: {e}", exc_info=True)
            self._is_running = False
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

def setup_argument_parser():
    '''Set up command line argument parser'''
    parser = argparse.ArgumentParser(
        description="SARS - Voice Assistant with Wake Word Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
        Examples:
        python main.py                                  # Use default settings
        python main.py --wake-words "hey sars" "sars"   # Custom wake words
        python main.py --energy 500                     # Lower energy threshold
        python main.py --list-mics                      # List available microphones (Linux)
        '''
    )
    
    
    # Audio configuration
    parser.add_argument(
        "--energy", 
        type=int, 
        default=AUDIO_CONFIG.energy_threshold,
        help="Energy threshold for microphone detection (default: %(default)s)"
    )
    parser.add_argument(
        "--record-timeout", 
        type=float, 
        default=AUDIO_CONFIG.record_timeout,
        help="Recording timeout in seconds (default: %(default)s)"
    )
    parser.add_argument(
        "--phrase-timeout", 
        type=float, 
        default=AUDIO_CONFIG.phrase_timeout,
        help="Phrase timeout in seconds (default: %(default)s)"
    )
    
    # Wake word configuration
    parser.add_argument(
        "--wake-words", 
        nargs='+', 
        default=WAKE_WORD_CONFIG.wake_words,
        help="Wake words to activate assistant (default: %(default)s)"
    )
    parser.add_argument(
        "--wake-timeout", 
        type=float, 
        default=WAKE_WORD_CONFIG.timeout_after_wake,
        help="Seconds to stay active after wake word (default: %(default)s)"
    )
    
    # Linux-specific options
    if 'linux' in platform:
        parser.add_argument(
            "--default-microphone", 
            default=SYSTEM_CONFIG.default_microphone,
            help="Default microphone name for Linux (default: %(default)s)"
        )
        parser.add_argument(
            "--list-mics", 
            action='store_true',
            help="List available microphones and exit"
        )
    
    return parser

def update_config_from_args(args):
    '''Update global configuration based on command line arguments'''
    
    # Update audio config
    AUDIO_CONFIG.energy_threshold = args.energy
    AUDIO_CONFIG.record_timeout = args.record_timeout
    AUDIO_CONFIG.phrase_timeout = args.phrase_timeout
    
    # Update wake word config
    WAKE_WORD_CONFIG.wake_words = args.wake_words
    WAKE_WORD_CONFIG.timeout_after_wake = args.wake_timeout
    
    # Linux-specific
    if 'linux' in platform:
        SYSTEM_CONFIG.default_microphone = args.default_microphone


async def async_main() -> None:
    """Asynchronous main entry point for the audio processor."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Handle special cases
    if 'linux' in platform and args.list_mics:
        from audio_handler import AudioHandler
        handler = AudioHandler()
        handler._list_microphones()
        return
    
    # Update configuration
    update_config_from_args(args)
    
    # Print startup information
    print("🤖 SARS Voice Assistant")
    print("=" * 40)
    print(f"📝 Model: {MODEL_CONFIG.model_name}")
    print(f"🎤 Energy Threshold: {AUDIO_CONFIG.energy_threshold}")
    print(f"⏱️ Record Timeout: {AUDIO_CONFIG.record_timeout}s")
    print(f"⏳ Phrase Timeout: {AUDIO_CONFIG.phrase_timeout}s")
    print(f"🎯 Wake Words: {', '.join(WAKE_WORD_CONFIG.wake_words)}")
    print(f"⏰ Wake Timeout: {WAKE_WORD_CONFIG.timeout_after_wake}s")
    print("=" * 40)
    print()
    
    # Create and start the audio processor service
    audio_processor = AudioProcessorService()
    
    try:
        await audio_processor.start(
            energy_threshold=AUDIO_CONFIG.energy_threshold,
            record_timeout=AUDIO_CONFIG.record_timeout,
            phrase_timeout=AUDIO_CONFIG.phrase_timeout,
            wake_words=WAKE_WORD_CONFIG.wake_words,
            timeout_after_wake=WAKE_WORD_CONFIG.timeout_after_wake
        )
        
        # Keep the service running until interrupted
        while audio_processor._is_running:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        logger.info("Audio processor was cancelled")
    except KeyboardInterrupt:
        logger.info("Shutting down audio processor...")
    except Exception as e:
        logger.error(f"Error in audio processor: {e}", exc_info=True)
        raise
    finally:
        await audio_processor.stop()
        logger.info("Audio processor stopped")

def main() -> None:
    """Synchronous entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(async_main())
        print("\n👋 Goodbye!")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(traceback.format_exc())
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()