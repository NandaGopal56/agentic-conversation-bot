'''Wake word detection functionality - ASYNC VERSION'''

import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional
from .config import WAKE_WORD_CONFIG

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

class WakeWordDetector:
    '''Handles wake word detection and activation state'''
    
    def __init__(self):
        self.wake_words = WAKE_WORD_CONFIG.wake_words
        self.confidence_threshold = WAKE_WORD_CONFIG.confidence_threshold
        self.timeout_duration = WAKE_WORD_CONFIG.timeout_after_wake
        
        self.is_active = False
        self.activation_time = None
        self.last_wake_detection = None
        self.last_speech_activity = None  # Track when speech was last detected
        
    async def check_for_wake_word(self, text: str) -> bool:
        '''
        Check if the transcribed text contains a wake word
        Returns True if wake word is detected
        '''
        if not text:
            return False
            
        text_lower = text.lower().strip()
        
        for wake_word in self.wake_words:
            if wake_word.lower() in text_lower:
                logger.info(f"Wake word detected: '{wake_word}'")
                await self._activate()
                return True
        
        return False
    
    async def _activate(self):
        '''Activate the assistant'''
        self.is_active = True
        self.activation_time = datetime.now()
        self.last_wake_detection = datetime.now()
        self.last_speech_activity = datetime.now()
        logger.info("Assistant activated! Listening for commands...")
    
    async def deactivate(self):
        '''Deactivate the assistant'''
        self.is_active = False
        self.activation_time = None
        self.last_speech_activity = None
        logger.info("Assistant deactivated. Waiting for wake word...")
    
    async def update_activity(self):
        '''Update activation state based on timeout - only when NOT speaking'''
        if not self.is_active:
            return
            
        if self.activation_time:
            reference_time = self.last_speech_activity or self.activation_time
            elapsed = datetime.now() - reference_time
            
            if elapsed.total_seconds() > self.timeout_duration:
                await self.deactivate()
    
    async def extend_activation(self):
        '''Extend the activation time when new speech is detected'''
        if self.is_active:
            current_time = datetime.now()
            self.activation_time = current_time
            self.last_speech_activity = current_time
    
    async def get_status(self) -> dict:
        '''Get current status of the wake word detector'''
        status = {
            'is_active': self.is_active,
            'wake_words': self.wake_words,
        }
        
        if self.is_active and self.activation_time:
            reference_time = self.last_speech_activity or self.activation_time
            elapsed = datetime.now() - reference_time
            remaining = max(0, self.timeout_duration - elapsed.total_seconds())
            status['time_remaining'] = remaining
            status['activation_time'] = self.activation_time
        
        return status
    
    async def detect_wake_word(self, audio_data: bytes) -> Optional[str]:
        '''Detect wake word in audio data'''
        if not audio_data:
            return None
            
        if len(audio_data) > 1000:  # Simple threshold check
            return self.wake_words[0] if self.wake_words else None
            
        return None
    
    async def should_process_speech(self) -> bool:
        '''Check if speech should be processed (assistant is active)'''
        return self.is_active
