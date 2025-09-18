'''Audio recording and processing functionality'''
import logging
import os
import asyncio
import speech_recognition as sr
from typing import Optional
from sys import platform
from .config import AUDIO_CONFIG, SYSTEM_CONFIG
import numpy as np

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/stt_audio_processor.log")]
)
logger = logging.getLogger(__name__)

class AudioHandler:
    '''Handles audio recording and preprocessing'''
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AudioHandler, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = AUDIO_CONFIG.energy_threshold
        self.recorder.dynamic_energy_threshold = AUDIO_CONFIG.dynamic_energy_threshold
        self.max_record_time = AUDIO_CONFIG.max_record_time
        
        self.data_queue = asyncio.Queue()
        self.loop = None   # will hold the main asyncio loop
        self._calibrated = False
        self._initialized = True

    def _get_microphone(self) -> sr.Microphone:
        '''Return a new microphone instance based on platform config'''
        if 'linux' in platform:
            mic_name = SYSTEM_CONFIG.default_microphone
            if not mic_name or mic_name == 'list':
                self._list_microphones()
                return None
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        return sr.Microphone(
                            sample_rate=AUDIO_CONFIG.sample_rate,
                            device_index=index
                        )
        return sr.Microphone(sample_rate=AUDIO_CONFIG.sample_rate)

    async def calibrate_microphone(self):
        '''Calibrate microphone for ambient noise'''
        if self._calibrated:
            return
            
        mic = self._get_microphone()
        if mic:
            with mic as src:
                logger.info("Calibrating microphone for ambient noise...")
                await asyncio.to_thread(
                    self.recorder.adjust_for_ambient_noise,
                    src
                )
            logger.info("Microphone calibrated.")
            self._calibrated = True

    def set_tts_audio(self, audio_bytes: bytes):
        """Provide TTS audio for echo suppression"""
        self.last_tts_audio = np.frombuffer(audio_bytes, dtype=np.int16)
    
    def _record_callback(self, _, audio: sr.AudioData):
        """Process audio and push into async queue"""
        try:
            mic_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            asyncio.run_coroutine_threadsafe(self.data_queue.put(mic_data.tobytes()), self.loop)
        except Exception as e:
            print(f"Error in callback: {e}")


    async def start_listening(self):
        """Start background listening (async)."""
        await self.calibrate_microphone()  # Auto-calibrate if not done
        
        self.loop = asyncio.get_running_loop()

        def _start_listening():
            mic = self._get_microphone()  # new mic just for listening
            return self.recorder.listen_in_background(
                mic,
                self._record_callback,
                phrase_time_limit=self.max_record_time,
            )

        return await asyncio.to_thread(_start_listening)

    
    async def get_audio_data(self) -> Optional[bytes]:
        '''Get accumulated audio data from async queue'''
        if self.data_queue.empty():
            return None

        audio_chunks = []
        while not self.data_queue.empty():
            chunk = await self.data_queue.get()
            audio_chunks.append(chunk)

        logger.info(f"Audio data chunks available: {len(audio_chunks)}")
        return b''.join(audio_chunks)

    
    async def clear_queue(self):
        """Clear the audio queue"""
        while not self.data_queue.empty():
            await self.data_queue.get()


# Create singleton instance and export
audio_handler = AudioHandler()

# Export the singleton instance  
__all__ = ['audio_handler', 'AudioHandler']