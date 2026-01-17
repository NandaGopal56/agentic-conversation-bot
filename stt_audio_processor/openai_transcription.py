import io
import wave
import numpy as np
from openai import OpenAI
from typing import Optional, Dict, Any
import asyncio
from .logger import logger
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def transcribe_audio_by_openai(
    audio_np: np.ndarray, 
    sample_rate: int = 16000,
    model: str = "gpt-4o-mini-transcribe"
) -> Optional[Dict[str, Any]]:
    """Convert numpy array to WAV in memory and send to OpenAI for transcription."""
    logger.info("Transcribing audio...")
    try:
        # Convert float32 to int16 if needed
        if audio_np.dtype == np.float32:
            audio_np = (audio_np * 32767).astype(np.int16)
        
        # Create in-memory WAV file
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)  # mono
                wf.setsampwidth(2)   # 2 bytes for int16
                wf.setframerate(sample_rate)
                wf.writeframes(audio_np.tobytes())
            
            # Reset buffer position to the beginning
            wav_buffer.seek(0)
            
            # BUG: Transcription is not happening strictly in the same language as spoken nor even in english with language or prompt explicitly specified
            # TODO: Fix this
            
            # Send to OpenAI
            transcription = await asyncio.to_thread(
                client.audio.transcriptions.create,
                model=model,
                file=("audio.wav", wav_buffer, "audio/wav"),
                response_format="json",
                language="en",
                prompt="You are a helpful assistant for transcribing audio. No matter what the audio is, you will always transcribe it to English. DO NOT transcribe it to any other language. Make sure to transcribe it to English. If you are not sure, transcribe it to English always.",
            )
            
            result = {
                'text': transcription.text.strip() if transcription.text else None,
                'confidence': getattr(transcription, 'confidence', 1.0)  # Not all models return confidence
            }

            logger.info(f"Transcription result: {result}")
            return result
            
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None