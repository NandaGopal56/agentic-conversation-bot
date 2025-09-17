import threading
import queue
import asyncio
from io import BytesIO
import simpleaudio as sa
from pydub import AudioSegment
# from stt_audio_processor.audio_handler import audio_handler_obj
from pydub import AudioSegment
import numpy as np


# Shared queue
audio_queue: "queue.Queue[AudioSegment]" = queue.Queue()

# Flag to stop playback
stop_playback = threading.Event()


def playback_worker():
    """Continuously play audio from queue."""
    while not stop_playback.is_set():
        try:
            segment = audio_queue.get(timeout=0.5)

            # Convert AudioSegment to raw PCM int16
            # tts_pcm = segment.set_channels(1).set_frame_rate(16000).get_array_of_samples()
            # tts_bytes = np.array(tts_pcm, dtype=np.int16).tobytes()
            # asyncio.run_coroutine_threadsafe(audio_handler_obj.set_tts_audio(tts_bytes), audio_handler_obj.loop)

        except queue.Empty:
            continue

        play_obj = sa.play_buffer(
            segment.raw_data,
            num_channels=segment.channels,
            bytes_per_sample=segment.sample_width,
            sample_rate=segment.frame_rate,
        )
        play_obj.wait_done()
        audio_queue.task_done()


# Start playback thread immediately when module is imported
threading.Thread(target=playback_worker, daemon=True).start()
