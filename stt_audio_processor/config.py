'''Configuration settings for the voice assistant'''

from dataclasses import dataclass
from typing import List

@dataclass
class AudioConfig:
    '''Audio processing configuration'''
    sample_rate: int = 16000
    energy_threshold: int = 1000
    max_record_time: float = 10.0 # Maximum recording time in seconds
    sentence_pause_timeout: float = 1 # seconds to wait for pause after sentence to consider it as end of sentence
    dynamic_energy_threshold: bool = False

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
WAKE_WORD_CONFIG = WakeWordConfig()
SYSTEM_CONFIG = SystemConfig()