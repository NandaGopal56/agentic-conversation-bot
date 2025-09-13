import logging
import numpy as np
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from .openai_transcription import transcribe_audio_by_openai

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Conversation manager that handles audio accumulation, transcription,
    and conversation tracking in a thread/message style structure.
    
    - Each conversation has a `thread_id` (like ChatGPT conversation ID).
    - Each transcribed message has its own `message_id`, `status`, and `text`.
    - Accumulated audio is cleared after a sentence is finalized.
    """

    def __init__(self):
        self.thread_id: str = self._new_thread_id()
        self.sentences: List[Dict[str, Any]] = []   # List of messages in current thread
        self.accumulated_audio: bytes = bytes()
        self.start_time: datetime = datetime.now()

    # ----------------------------
    # Internal Helpers
    # ----------------------------
    def _new_thread_id(self) -> str:
        """Generate a new thread identifier (conversation id)."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _new_message_id(self) -> str:
        """Generate a unique message identifier."""
        return str(uuid.uuid4())

    # ----------------------------
    # Audio Handling
    # ----------------------------
    async def accumulate_audio(self, audio_data: bytes) -> None:
        """
        Accumulate incoming raw audio bytes into the buffer.
        
        Args:
            audio_data (bytes): Raw PCM audio data.
        """
        self.accumulated_audio += audio_data

    async def clear_audio_buffer(self) -> None:
        """Clear the accumulated audio buffer."""
        self.accumulated_audio = bytes()

    # ----------------------------
    # Transcription Handling
    # ----------------------------
    async def transcribe_audio(self, audio_np: np.ndarray) -> Optional[str]:
        """
        Transcribe audio data using OpenAI model.
        
        Args:
            audio_np (np.ndarray): Audio data as float32 numpy array.
        
        Returns:
            Optional[str]: Transcribed text or None if empty/error.
        """
        if len(audio_np) == 0:
            return None
        try:
            result = await transcribe_audio_by_openai(audio_np)
            text = result.get("text", "").strip()
            return text if text else None
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    async def transcribe_accumulated_audio(self) -> Optional[str]:
        """
        Transcribe the accumulated audio buffer into a completed sentence.
        
        Returns:
            Optional[str]: Finalized sentence text.
        """
        if not self.accumulated_audio:
            return None

        audio_np = np.frombuffer(self.accumulated_audio, dtype=np.int16).astype(np.float32) / 32768.0
        text = await self.transcribe_audio(audio_np)
        return text

    async def add_completed_sentence(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Add a completed sentence into the conversation thread as a message.
        
        Args:
            text (str): Finalized transcribed text.
        
        Returns:
            Optional[Dict[str, Any]]: Message payload if text was valid, else None.
        """
        if not text.strip():
            return None

        message = {
            "message_id": self._new_message_id(),
            "id": len(self.sentences) + 1,
            "text": text.strip(),
            "timestamp": datetime.now().isoformat(),
            "status": "unprocessed"
        }
        self.sentences.append(message)

        # Clear buffer for next message
        self.accumulated_audio = bytes()
        return message

    # ----------------------------
    # Session Management
    # ----------------------------
    async def start_new_session(self) -> str:
        """
        Start a new conversation session (thread).
        
        Returns:
            str: New thread_id.
        """
        self.thread_id = self._new_thread_id()
        self.sentences.clear()
        self.accumulated_audio = bytes()
        self.start_time = datetime.now()
        return self.thread_id

    async def end_current_session(self) -> None:
        """End the current session by clearing remaining audio."""
        self.accumulated_audio = bytes()

    async def clear(self) -> None:
        """Completely clear the session (sentences + buffer)."""
        self.sentences.clear()
        self.accumulated_audio = bytes()

    # ----------------------------
    # Event / Payload
    # ----------------------------
    async def get_conversation_event(self) -> Dict[str, Any]:
        """
        Get the complete structured conversation event.
        
        Returns:
            Dict[str, Any]: Event payload with thread and message history.
        """
        return {
            "thread_id": self.thread_id,
            "start_time": self.start_time.isoformat(),
            "sentence_count": len(self.sentences),
            "sentences": self.sentences.copy(),
            "status": "active" if self.accumulated_audio else "idle"
        }
