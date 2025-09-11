'''Transcription management and conversation handling (async)'''
import logging
import numpy as np
from datetime import datetime
from typing import Optional
from .openai_transcription import transcribe_audio_by_openai

logger = logging.getLogger(__name__)

class ConversationSession:
    '''Single conversation session with sentences'''
    
    def __init__(self):
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.sentences = []
        self.accumulated_audio = bytes()  # Audio buffer for current sentence
        self.start_time = datetime.now()
    
    async def add_audio(self, audio_data: bytes):
        '''Add audio to current sentence buffer'''
        self.accumulated_audio += audio_data
    
    async def add_completed_sentence(self, text: str):
        '''Add a completed sentence'''
        if text.strip():
            sentence = {
                'text': text.strip(),
                'timestamp': datetime.now(),
                'id': len(self.sentences) + 1
            }
            self.sentences.append(sentence)
            # Clear audio buffer
            self.accumulated_audio = bytes()
            return sentence
        return None
    
    async def clear_audio_buffer(self):
        '''Clear the audio buffer'''
        self.accumulated_audio = bytes()
    
    async def clear(self):
        '''Clear session'''
        self.sentences.clear()
        self.accumulated_audio = bytes()


class TranscriptionManager:
    '''Manages transcription and sessions'''
    
    def __init__(self):
        self.current_session = ConversationSession()
    
    async def transcribe_audio(self, audio_np) -> Optional[str]:
        '''Transcribe audio'''
        if len(audio_np) == 0:
            return None
        try:
            result = await transcribe_audio_by_openai(audio_np)
            return result['text'].strip() if result['text'].strip() else None
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    async def accumulate_audio(self, audio_data: bytes):
        '''Accumulate audio without transcribing'''
        await self.current_session.add_audio(audio_data)
    
    async def transcribe_accumulated_audio(self) -> Optional[str]:
        '''Transcribe all accumulated audio as one sentence'''
        if len(self.current_session.accumulated_audio) == 0:
            return None
            
        # Convert to numpy
        audio_np = np.frombuffer(
            self.current_session.accumulated_audio, 
            dtype=np.int16
        ).astype(np.float32) / 32768.0
        
        # Transcribe the complete sentence
        text = await self.transcribe_audio(audio_np)
        return text
    
    async def add_completed_sentence(self, text: str):
        '''Add a completed sentence to session'''
        return await self.current_session.add_completed_sentence(text)
    
    async def start_new_session(self) -> str:
        '''Start new session'''
        self.current_session = ConversationSession()
        return self.current_session.session_id
    
    async def end_current_session(self):
        '''End current session'''
        # Clear any remaining audio buffer
        await self.current_session.clear_audio_buffer()
    
    async def get_conversation_event(self) -> dict:
        '''Get complete conversation event structure'''
        return {
            'session_id': self.current_session.session_id,
            'start_time': self.current_session.start_time,
            'sentence_count': len(self.current_session.sentences),
            'sentences': self.current_session.sentences.copy(),  # Return copy of sentences
            'status': 'active' if len(self.current_session.accumulated_audio) > 0 else 'idle'
        }
