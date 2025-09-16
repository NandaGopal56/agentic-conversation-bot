'''Main voice assistant coordinator'''

import logging
import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional

from langchain_core.callbacks import file

from .audio_handler import AudioHandler
from .wake_word_detector import WakeWordDetector
from .conversation_manager import ConversationManager
from .config import AUDIO_CONFIG

from communication_bus.inmemory_bus import InMemoryBus


# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/stt_audio_processor.log")]
)
logger = logging.getLogger(__name__)


class VoiceProcessor:
    '''Main voice assistant that coordinates all components'''
    
    def __init__(self, bus: InMemoryBus):
        self.bus = bus
        self.audio_handler = AudioHandler()
        self.wake_word_detector = WakeWordDetector()
        self.conversation_manager = ConversationManager()
        
        # State management
        self.last_audio_time: Optional[datetime] = None
        self.sentence_pause_timeout = AUDIO_CONFIG.sentence_pause_timeout
        self.is_running = False
        self.is_speaking = False


    async def _display_status(self):
        '''Display simple status'''
        wake_status = await self.wake_word_detector.get_status()
        if wake_status['is_active']:
            status = "LISTENING" if not self.is_speaking else "SPEAKING"
            remaining = wake_status.get('time_remaining', 0)
            logger.info(f"Status: {status} ({remaining:.0f}s)")
        else:
            wake_words = ' or '.join(wake_status['wake_words'])
            logger.info(f"Status: SLEEPING - Say '{wake_words}' to wake")
        
        conversation_event = await self.conversation_manager.get_conversation_event()
        logger.info(conversation_event)

        await self.bus.publish("voice/commands", conversation_event)
    
    async def _process_transcription(self, text: str, sentence_complete: bool):
        '''Process transcribed text'''
        if not self.wake_word_detector.is_active:
            if await self.wake_word_detector.check_for_wake_word(text):
                await self.conversation_manager.start_new_session()
            return
        
        await self.wake_word_detector.extend_activation()
        
        if sentence_complete:
            await self.conversation_manager.add_completed_sentence(text)
            self.is_speaking = False
            await self._display_status()
    
    async def run(self):
        '''Main run loop (async)'''        
        await self.audio_handler.start_listening()
        self.is_running = True
        
        try:
            while self.is_running:
                # running an infinite loop to check for audio data and get the audio data store in a queue and log the time
                current_time = datetime.now()
                audio_data = await self.audio_handler.get_audio_data()
                logger.debug(f'Audio data available: {len(audio_data) if audio_data else 0}')
                
                # if audio data is available, mark the user as speaking, update the last audio time,
                # and accumulate the audio data in the conversation manager
                if audio_data:
                    logger.debug('User is speaking')
                    # User is speaking
                    self.last_audio_time = current_time
                    self.is_speaking = True
                    await self.conversation_manager.accumulate_audio(audio_data)
                    logger.debug('Accumulated audio')

                # if no audio data is available, check for pause long enough to complete a sentence
                else:
                    logger.debug('No audio, check for pause long enough to complete a sentence')
                    # No audio, check for pause long enough to complete a sentence

                    # if the user is speaking and the last audio time is available and the pause is long enough to complete a sentence,
                    # transcribe the accumulated audio and process the transcription
                    if (self.is_speaking and self.last_audio_time and
                        current_time - self.last_audio_time > timedelta(seconds=self.sentence_pause_timeout)):
                        logger.debug('Pause long enough to complete a sentence')
                        
                        text = await self.conversation_manager.transcribe_accumulated_audio()
                        logger.debug(f'Transcribed text: {text}')
                        if text:
                            await self._process_transcription(text, sentence_complete=True)
                            logger.debug('Processed transcription')

                        # Reset everything so transcription won't repeat
                        self.is_speaking = False
                        self.last_audio_time = None
                        await self.conversation_manager.end_current_session()
                        logger.debug('Reset everything')
                    
                    # if the user is not speaking, check for wake word
                    logger.debug('No audio, check for wake word')
                    logger.debug(f'Is speaking: {self.is_speaking}')
                    logger.debug(f'Last audio time: {self.last_audio_time}')
                    logger.debug(f'Current time: {current_time}')
                    logger.debug(f'Pause timeout: {self.sentence_pause_timeout}')
                    
                    # Handle wake word state
                    was_active = self.wake_word_detector.is_active
                    logger.debug(f'Wake word state: {was_active}')

                    if not self.is_speaking:
                        await self.wake_word_detector.update_activity()
                        logger.debug('Updated wake word activity')
                    else:
                        await self.wake_word_detector.extend_activation()
                        logger.debug('Extended wake word activation')
                    
                    # if the wake word state has changed, update the conversation manager to end the current session and reset the speaking state
                    logger.debug(f'Wake word state changed: {was_active} -> {self.wake_word_detector.is_active}')
                    
                    if was_active != self.wake_word_detector.is_active:
                        if not self.wake_word_detector.is_active:
                            await self.conversation_manager.end_current_session()
                            self.is_speaking = False
                        await self._display_status()
                    
                    # sleep for a short time to avoid busy waiting
                    logger.debug('Sleeping for a short time to avoid busy waiting')
                    await asyncio.sleep(0.1)
                        
        except asyncio.CancelledError:
            await self.stop()

    
    async def stop(self):
        '''Stop the voice assistant'''
        logger.info("Stopping...")
        self.is_running = False
        logger.info("Goodbye")
