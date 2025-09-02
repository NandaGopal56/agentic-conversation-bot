'''Main voice assistant coordinator (async version)'''

import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional

from .audio_handler import AudioHandler
from .wake_word_detector import WakeWordDetector
from .transcription_manager import TranscriptionManager

from communication_bus.inmemory_bus import InMemoryBus

class VoiceProcessor:
    '''Main voice assistant that coordinates all components (async)'''
    
    def __init__(self):
        self.audio_handler = AudioHandler()
        self.wake_word_detector = WakeWordDetector()
        self.transcription_manager = TranscriptionManager()
        
        # State management
        self.last_audio_time: Optional[datetime] = None
        self.sentence_pause_timeout = 2.5
        self.is_running = False
        self.is_speaking = False

    async def _display_status(self):
        '''Display simple status'''
        wake_status = await self.wake_word_detector.get_status()
        if wake_status['is_active']:
            status = "LISTENING" if not self.is_speaking else "SPEAKING"
            remaining = wake_status.get('time_remaining', 0)
            print(f"Status: {status} ({remaining:.0f}s)")
        else:
            wake_words = ' or '.join(wake_status['wake_words'])
            print(f"Status: SLEEPING - Say '{wake_words}' to wake")
        
        conversation_event = await self.transcription_manager.get_conversation_event()
        print(conversation_event)

        # bus = InMemoryBus()
        # await bus.connect()
        # await bus.publish("voice/commands", conversation_event)
        
        print()
    
    async def _process_transcription(self, text: str, sentence_complete: bool):
        '''Process transcribed text'''
        if not self.wake_word_detector.is_active:
            if await self.wake_word_detector.check_for_wake_word(text):
                await self.transcription_manager.start_new_session()
            return
        
        await self.wake_word_detector.extend_activation()
        
        if sentence_complete:
            await self.transcription_manager.add_completed_sentence(text)
            self.is_speaking = False
            print()
            await self._display_status()
    
    async def run(self):
        '''Main run loop (async)'''        
        await self.audio_handler.start_listening()
        self.is_running = True
        await self._display_status()
        
        try:
            while self.is_running:
                current_time = datetime.now()
                audio_data = await self.audio_handler.get_audio_data()
                
                if audio_data:
                    self.last_audio_time = current_time
                    self.is_speaking = True
                    await self.transcription_manager.accumulate_audio(audio_data)
                
                else:
                    if (self.is_speaking and self.last_audio_time and 
                        current_time - self.last_audio_time > timedelta(seconds=self.sentence_pause_timeout)):
                        
                        text = await self.transcription_manager.transcribe_accumulated_audio()
                        if text:
                            await self._process_transcription(text, sentence_complete=True)
                    
                    was_active = self.wake_word_detector.is_active
                    
                    if not self.is_speaking:
                        await self.wake_word_detector.update_activity()
                    else:
                        await self.wake_word_detector.extend_activation()
                    
                    if was_active != self.wake_word_detector.is_active:
                        if not self.wake_word_detector.is_active:
                            await self.transcription_manager.end_current_session()
                            self.is_speaking = False
                        await self._display_status()
                    
                    await asyncio.sleep(0.1)
                        
        except asyncio.CancelledError:
            await self.stop()
    
    async def stop(self):
        '''Stop the voice assistant'''
        print("Stopping...")
        self.is_running = False
        print("Goodbye")
