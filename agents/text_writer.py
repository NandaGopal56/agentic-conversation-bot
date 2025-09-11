import asyncio
from communication_bus.inmemory_bus import bus




paragraph = f"""
    The first move is what sets everything in motion. 
    Every step that follows is guided by that initial decision. 
    In life, just like in chess, momentum is everything.
    A small misstep can throw off your entire strategy.
    We must be prepared for the unexpected.
    """

def stream_words(text, chunk_size=3):
    """Yield text in chunks of whole words (simulate LLM streaming)."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

async def stream_response():
    await bus.connect()
    # Simulate LLM streaming words
    for chunk in stream_words(paragraph, chunk_size=3):
        print(f"LLM streamed: {chunk}")
        await bus.publish("voice/commands/llm_response", {"text": chunk})
        await asyncio.sleep(0.5)  # simulate LLM streaming delay

async def main():
    await stream_response()
    