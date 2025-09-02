'''Audio recording and processing functionality'''

import numpy as np
import speech_recognition as sr
from queue import Queue
from typing import Optional, Callable
from sys import platform
from config import AUDIO_CONFIG, SYSTEM_CONFIG

class AudioHandler:
    '''Handles audio recording and preprocessing'''
    
    def __init__(self):
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = AUDIO_CONFIG.energy_threshold
        self.recorder.dynamic_energy_threshold = AUDIO_CONFIG.dynamic_energy_threshold
        
        self.data_queue = Queue()
        self.source = self._setup_microphone()
        self._calibrate_microphone()
        
    def _setup_microphone(self) -> sr.Microphone:
        '''Setup microphone based on platform'''
        if 'linux' in platform:
            mic_name = SYSTEM_CONFIG.default_microphone
            if not mic_name or mic_name == 'list':
                self._list_microphones()
                return None
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        return sr.Microphone(sample_rate=AUDIO_CONFIG.sample_rate, device_index=index)
        else:
            return sr.Microphone(sample_rate=AUDIO_CONFIG.sample_rate)
    
    def _list_microphones(self):
        '''List available microphones'''
        print("Available microphone devices are: ")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Microphone with name \"{name}\" found")
    
    def _calibrate_microphone(self):
        '''Calibrate microphone for ambient noise'''
        if self.source:
            with self.source:
                print("Calibrating microphone for ambient noise...")
                self.recorder.adjust_for_ambient_noise(self.source)
                print("Microphone calibrated.")
    
    def _record_callback(self, _, audio: sr.AudioData) -> None:
        '''Callback function for background recording'''
        data = audio.get_raw_data()
        self.data_queue.put(data)
    
    def start_listening(self, callback: Optional[Callable] = None):
        '''Start background listening'''
        if not self.source:
            raise RuntimeError("No microphone source available")
        
        actual_callback = callback or self._record_callback
        self.recorder.listen_in_background(
            self.source, 
            actual_callback, 
            phrase_time_limit=AUDIO_CONFIG.record_timeout
        )
    
    def get_audio_data(self) -> Optional[bytes]:
        '''Get accumulated audio data from queue'''
        if self.data_queue.empty():
            return None
        
        # Combine all audio data in queue
        audio_chunks = []
        while not self.data_queue.empty():
            audio_chunks.append(self.data_queue.get())
        
        return b''.join(audio_chunks)
    
    def audio_to_numpy(self, audio_data: bytes) -> np.ndarray:
        '''Convert raw audio bytes to numpy array for whisper'''
        return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    def clear_queue(self):
        '''Clear the audio queue'''
        while not self.data_queue.empty():
            self.data_queue.get()