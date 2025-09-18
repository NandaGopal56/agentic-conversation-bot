import threading
import queue
import simpleaudio as sa
from pydub import AudioSegment

# Shared queue
audio_queue: "queue.Queue[AudioSegment]" = queue.Queue()

# Flag to stop playback
stop_playback = threading.Event()


def playback_worker():
    """Continuously play audio from queue."""
    while not stop_playback.is_set():
        try:
            segment = audio_queue.get(timeout=0.5)
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
