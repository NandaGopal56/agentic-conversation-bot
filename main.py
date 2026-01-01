from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
import signal

from stt_audio_processor.stt_processor import STTAudioProcessorService
from tts_audio_processor.tts_processor import TTSAudioProcessorService
from agents.agent_processor import AgentProcessor
from live_interaction.ui_service import UIService

# demo code
from communication_bus.inmemory_bus import bus
from agents.bot import invoke_conversation

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Extra async work (your custom logic)
# -------------------------------------------------------------------
async def other_operations():
    """
    Any additional async operations you want to run
    after services are started.
    """
    # while True:
    #     try:
    #         user_input = input("You: ").strip()
    #     except (EOFError, KeyboardInterrupt):
    #         break

    #     if not user_input:
    #         continue
    #     if user_input.lower() in {"exit", "quit"}:
    #         break

    #     print("Assistant: ", end="", flush=True)

    #     async for token in invoke_conversation(user_input, thread_id=13):
    #         print(token, end="", flush=True)

    #     print("\n")
    pass


# -------------------------------------------------------------------
# Main async entry point
# -------------------------------------------------------------------
async def async_main():
    logger.info("Starting Agentic Conversation Bot...")

    # Initialize services
    # stt_audio_processor = STTAudioProcessorService()
    # tts_audio_processor = TTSAudioProcessorService()
    agent_processor = AgentProcessor()
    ui_service = UIService()

    # Graceful shutdown event
    shutdown_event = asyncio.Event()

    # ---------------------------------------------------------------
    # Signal handling
    # ---------------------------------------------------------------
    def _shutdown():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, _shutdown)
    loop.add_signal_handler(signal.SIGTERM, _shutdown)

    try:
        # Start services
        await asyncio.gather(
            # stt_audio_processor.start(),
            # tts_audio_processor.start(),
            agent_processor.start(),
            ui_service.start(),
        )

        logger.info("All services started")

        # Run background tasks
        background_tasks = [
            asyncio.create_task(other_operations()),
        ]

        # Keep app alive until shutdown
        await shutdown_event.wait()

    except Exception:
        logger.exception("Unhandled error")

    finally:
        logger.info("Stopping services...")

        # Cancel background tasks
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

        await asyncio.gather(
            # stt_audio_processor.stop(),
            # tts_audio_processor.stop(),
            agent_processor.stop(),
            ui_service.stop(),
            return_exceptions=True,
        )

        logger.info("Agentic Conversation Bot shut down cleanly")


# -------------------------------------------------------------------
# Sync entry point
# -------------------------------------------------------------------
def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception:
        logger.exception("Fatal error")


if __name__ == "__main__":
    main()
