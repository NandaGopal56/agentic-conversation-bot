
from communication_bus.inmemory_bus import bus
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

async def write_response_to_bus(payload: Dict[str, Any]):
    '''
    This function is used to write the response to the bus
    '''
    try:
        await bus.connect()
        await bus.publish("voice/commands/llm_response", payload)
    except Exception as e:
        logger.error(f"Error writing response to bus: {e}", exc_info=True)
