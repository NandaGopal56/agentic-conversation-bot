import os
import asyncio
from io import BytesIO
from dotenv import load_dotenv
from pydub import AudioSegment
import simpleaudio as sa
from elevenlabs.client import ElevenLabs

load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_TTS_ONLY_API_KEY"))

dummy_text = (
    "The first move is what sets everything in motion. "
    "Every step that follows is guided by that initial decision. "
    "In life, just like in chess, momentum is everything."
)

def stream_words(text, chunk_size=3):
    """Yield text in chunks of whole words (simulate LLM streaming)."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

async def tts_task(chunk: str, queue: asyncio.Queue):
    """Convert chunk to TTS and push AudioSegment into queue."""
    if not chunk.strip():
        return
    
    audio = client.text_to_speech.convert(
        text=chunk,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    audio_bytes = b"".join(audio)
    audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
    
    await queue.put(audio_segment)

async def playback_worker(queue: asyncio.Queue):
    """Continuously stitch audio chunks and play them in order."""
    while True:
        segment = await queue.get()
        if segment is None:  # sentinel to stop
            break
        play_obj = sa.play_buffer(
            segment.raw_data,
            num_channels=segment.channels,
            bytes_per_sample=segment.sample_width,
            sample_rate=segment.frame_rate,
        )
        play_obj.wait_done()
        queue.task_done()

async def main():
    queue = asyncio.Queue()

    # Start playback worker
    playback = asyncio.create_task(playback_worker(queue))

    # Simulate LLM streaming words
    for chunk in stream_words(dummy_text, chunk_size=3):
        print(f"LLM streamed: {chunk}")
        asyncio.create_task(tts_task(chunk, queue))
        await asyncio.sleep(0.5)  # simulate LLM streaming delay

    # Wait for all tasks to finish
    await queue.join()
    await queue.put(None)  # stop playback worker
    await playback

asyncio.run(main())
