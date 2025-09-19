'''Audio recording and processing functionality'''
import asyncio
import speech_recognition as sr
from typing import Optional
from sys import platform
from .config import AUDIO_CONFIG, SYSTEM_CONFIG
import numpy as np
import threading
from .logger import logger


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
        
        self._stop_listening = None
        self._listening_lock = threading.Lock()
        self.muted = False
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

    def mute(self):
        """Stop the background listener so _record_callback will not be called."""
        with self._listening_lock:
            if self._stop_listening is None:
                # already muted / not listening
                self.muted = True
                logger.debug("Mute requested but listener already stopped")
                return
            try:
                # some listen_in_background stop functions accept a kwarg, some don't
                self._stop_listening(wait_for_stop=False)
            except TypeError:
                try:
                    self._stop_listening()
                except Exception:
                    pass
            self._stop_listening = None
            self.muted = True
            logger.info("Microphone listener stopped (muted)")

    def unmute(self):
        """Restart background listener."""
        with self._listening_lock:
            if self._stop_listening is not None:
                self.muted = False
                logger.debug("Unmute requested but already listening")
                return
            mic = self._get_microphone()
            if mic is None:
                raise RuntimeError("No microphone available to unmute")
            self._stop_listening = self.recorder.listen_in_background(
                mic,
                self._record_callback,
                phrase_time_limit=self.max_record_time,
            )
            self.muted = False
            logger.info("Microphone listener restarted (unmuted)")

    def set_tts_audio(self, audio_bytes: bytes):
        """Provide TTS audio for echo suppression"""
        self.last_tts_audio = np.frombuffer(audio_bytes, dtype=np.int16)
    
    def _record_callback(self, _, audio: sr.AudioData):
        """Process audio and push into queue"""
        try:
            if self.muted:
                logger.info("Microphone muted: dropping audio frame")
                return
            logger.info("Microphone unmuted: processing audio frame")

            mic_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            self.data_queue.put_nowait(mic_data.tobytes())   # works fine from any thread
        except Exception as e:
            logger.error(f"Error in callback: {e}")


    async def start_listening(self):
        """Start background listening (always safe to call)."""
        await self.calibrate_microphone()

        def _start():
            with self._listening_lock:
                if self._stop_listening is not None:
                    return  # already listening
                mic = self._get_microphone()
                if mic is None:
                    raise RuntimeError("No microphone available")
                self._stop_listening = self.recorder.listen_in_background(
                    mic,
                    self._record_callback,
                    phrase_time_limit=self.max_record_time,
                )
                self.muted = False
                logger.info("Background listening started")

        await asyncio.to_thread(_start)

    
    async def get_audio_data(self) -> Optional[bytes]:
        """Async: drain audio queue into one bytes object"""
        def _drain():
            if self.data_queue.empty():
                return None
            chunks = []
            while not self.data_queue.empty():
                chunks.append(self.data_queue.get_nowait())
            return b''.join(chunks)

        return await asyncio.to_thread(_drain)

    
    async def clear_queue(self):
        """Clear the audio queue"""
        while not self.data_queue.empty():
            await self.data_queue.get()


# Create singleton instance and export
audio_handler = AudioHandler()

# Export the singleton instance  
__all__ = ['audio_handler', 'AudioHandler']