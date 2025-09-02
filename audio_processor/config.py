'''Configuration settings for the voice assistant'''

import os
from dataclasses import dataclass
from typing import List

@dataclass
class AudioConfig:
    '''Audio processing configuration'''
    sample_rate: int = 16000
    energy_threshold: int = 1000
    record_timeout: float = 2.0
    phrase_timeout: float = 3.0
    dynamic_energy_threshold: bool = False

@dataclass
class ModelConfig:
    '''Whisper model configuration'''
    model_size: str = "small"
    language: str = "en" 

    @property
    def model_name(self) -> str:
        '''Get the full model name with language suffix'''
        return f"{self.model_size}.{self.language}"

@dataclass
class WakeWordConfig:
    '''Wake word detection configuration'''
    wake_words: List[str] = None
    confidence_threshold: float = 0.7
    timeout_after_wake: float = 30.0  # seconds to stay active after wake word
    
    def __post_init__(self):
        if self.wake_words is None:
            self.wake_words = ["alexa"]

@dataclass
class SystemConfig:
    '''System-wide configuration'''
    default_microphone: str = "pulse"  # for Linux
    

# Global configuration instances
AUDIO_CONFIG = AudioConfig()
MODEL_CONFIG = ModelConfig()
WAKE_WORD_CONFIG = WakeWordConfig()
SYSTEM_CONFIG = SystemConfig()