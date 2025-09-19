from elevenlabs.client import ElevenLabs
import os
from dotenv import load_dotenv
from io import BytesIO
from pydub import AudioSegment
from .logger import logger

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_TTS_ONLY_API_KEY"))

def tts_generate_audio(text: str) -> AudioSegment:
    """Convert text to speech and return as AudioSegment."""
    logger.info(f"Generating TTS audio for text: {text}")
    try:
        audio = client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        audio_bytes = b"".join(audio)
        logger.info(f"Generated TTS audio for text: {text}")

        return AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
    
    except Exception as e:
        logger.error(f"Failed to generate TTS audio: {e}")
        return None
