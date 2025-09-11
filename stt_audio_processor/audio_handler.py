'''Audio recording and processing functionality'''
import logging
import asyncio
import numpy as np
import speech_recognition as sr
from typing import Optional
from sys import platform
from .config import AUDIO_CONFIG, SYSTEM_CONFIG

logger = logging.getLogger(__name__)

class AudioHandler:
    '''Handles audio recording and preprocessing'''
    
    def __init__(self):
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = AUDIO_CONFIG.energy_threshold
        self.recorder.dynamic_energy_threshold = AUDIO_CONFIG.dynamic_energy_threshold
        self.max_record_time = AUDIO_CONFIG.max_record_time
        
        self.data_queue = asyncio.Queue()
        self.loop = None   # will hold the main asyncio loop

        # schedule calibration in background
        asyncio.create_task(self._calibrate_microphone())

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

    async def _calibrate_microphone(self):
        '''Calibrate microphone for ambient noise'''
        mic = self._get_microphone()
        if mic:
            with mic as src:
                logger.info("Calibrating microphone for ambient noise...")
                await asyncio.to_thread(
                self.recorder.adjust_for_ambient_noise,
                src
            )
            logger.info("Microphone calibrated.")

    
    def _record_callback(self, _, audio: sr.AudioData) -> None:
        """Callback function for background recording (sync)."""
        try:
            data = audio.get_raw_data()
            # Schedule coroutine to push into async queue
            asyncio.run_coroutine_threadsafe(self.data_queue.put(data), self.loop)
        except Exception as e:
            logger.error(f"Error in callback: {e}")


    async def start_listening(self):
        """Start background listening (async)."""
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

        return b''.join(audio_chunks)

    
    async def audio_to_numpy(self, audio_data: bytes) -> np.ndarray:
        '''Convert raw audio bytes to numpy array for whisper'''
        return await asyncio.to_thread(
            lambda: np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
    
    async def clear_queue(self):
        """Clear the audio queue"""
        while not self.data_queue.empty():
            await self.data_queue.get()
