import asyncio
import logging
from communication_bus.inmemory_bus import InMemoryBus, bus
from .receiver import on_voice_command

logger = logging.getLogger(__name__)

class AgentService:
    """Service for managing agent lifecycle and message handling."""

    def __init__(self):
        self.bus: InMemoryBus = bus
        self._is_running = False
        self._run_task: asyncio.Task | None = None

    async def _run(self):
        """Main agent loop (extend later for AI processing)."""
        while self._is_running:
            await asyncio.sleep(0.5)  # placeholder loop
            # could add periodic work here if needed
        logger.info("Agent loop stopped")

    async def start(self, **kwargs) -> None:
        """Start the agent service asynchronously."""
        if self._is_running:
            logger.warning("Agent service already running")
            return

        try:
            logger.info("Starting Agent Service...")
            await self.bus.connect()
            self.bus.subscribe("voice/commands", on_voice_command)

            self._is_running = True
            self._run_task = asyncio.create_task(self._run())
            logger.info("Agent Service started")

        except Exception as e:
            logger.error(f"Failed to start agent service: {e}", exc_info=True)
            self._is_running = False
            await self.bus.disconnect()
            raise

    async def stop(self) -> None:
        """Stop the agent service and clean up resources."""
        if not self._is_running:
            return

        logger.info("Stopping Agent Service...")
        self._is_running = False

        if self._run_task:
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass

        await self.bus.disconnect()
        logger.info("Agent Service stopped")