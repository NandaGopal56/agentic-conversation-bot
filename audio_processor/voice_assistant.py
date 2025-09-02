'''Main voice assistant coordinator'''

import os
import time
from datetime import datetime, timedelta
from typing import Optional

from audio_handler import AudioHandler
from wake_word_detector import WakeWordDetector
from transcription_manager import TranscriptionManager

from communication_bus.inmemory_bus import InMemoryBus

class VoiceProcessor:
    '''Main voice assistant that coordinates all components'''
    
    def __init__(self):
        self.audio_handler = AudioHandler()
        self.wake_word_detector = WakeWordDetector()
        self.transcription_manager = TranscriptionManager()
        
        # State management
        self.last_audio_time = None
        self.sentence_pause_timeout = 2.5
        self.is_running = False
        self.is_speaking = False

    def _display_status(self):
        '''Display simple status'''
        # Current status
        wake_status = self.wake_word_detector.get_status()
        if wake_status['is_active']:
            status = "LISTENING" if not self.is_speaking else "SPEAKING"
            remaining = wake_status.get('time_remaining', 0)
            print(f"Status: {status} ({remaining:.0f}s)")
        else:
            wake_words = ' or '.join(wake_status['wake_words'])
            print(f"Status: SLEEPING - Say '{wake_words}' to wake")
        
        # Show conversation event
        conversation_event = self.transcription_manager.get_conversation_event()
        print(conversation_event)

        bus = InMemoryBus()
        bus.connect()
        bus.publish("voice/commands", conversation_event)
        
        print()
    
    def _process_transcription(self, text: str, sentence_complete: bool):
        '''Process transcribed text'''
        # Check for wake word if sleeping
        if not self.wake_word_detector.is_active:
            if self.wake_word_detector.check_for_wake_word(text):
                self.transcription_manager.start_new_session()
            return
        
        # Extend wake time
        self.wake_word_detector.extend_activation()
        
        # Only process completed sentences
        if sentence_complete:
            self.transcription_manager.add_completed_sentence(text)
            self.is_speaking = False
            print()
            self._display_status()
    
    def run(self):
        '''Main run loop'''
        if not self.audio_handler.source:
            print("No microphone available")
            return
        
        self.audio_handler.start_listening()
        self.is_running = True
        self._display_status()
        
        try:
            while self.is_running:
                current_time = datetime.now()
                audio_data = self.audio_handler.get_audio_data()
                
                if audio_data:
                    # Audio detected - just accumulate, don't transcribe yet
                    self.last_audio_time = current_time
                    self.is_speaking = True
                    
                    # Just store audio, don't transcribe until pause
                    self.transcription_manager.accumulate_audio(audio_data)
                
                else:
                    # Check for sentence completion due to silence
                    if (self.is_speaking and self.last_audio_time and 
                        current_time - self.last_audio_time > timedelta(seconds=self.sentence_pause_timeout)):
                        
                        # Transcribe the complete sentence now
                        text = self.transcription_manager.transcribe_accumulated_audio()
                        if text:
                            self._process_transcription(text, sentence_complete=True)
                    
                    # Update wake detector - but don't timeout if still speaking
                    was_active = self.wake_word_detector.is_active
                    
                    # Only update activity if not currently speaking
                    if not self.is_speaking:
                        self.wake_word_detector.update_activity()
                    else:
                        # If speaking, extend activation to prevent timeout
                        self.wake_word_detector.extend_activation()
                    
                    if was_active != self.wake_word_detector.is_active:
                        if not self.wake_word_detector.is_active:
                            self.transcription_manager.end_current_session()
                            self.is_speaking = False
                        self._display_status()
                    
                    time.sleep(0.1)
                        
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        '''Stop the voice assistant'''
        print("Stopping...")
        self.is_running = False
        print("Goodbye")