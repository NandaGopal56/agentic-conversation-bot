"""
Agents module for the Agetic Conversation Bot.

This module contains the core agent implementations that process voice commands
and manage conversations.
"""
from typing import Dict, Any
import logging
import asyncio
from .bot import invoke_conversation

logger = logging.getLogger(__name__)

def on_voice_command(topic: str, payload: Dict[str, Any]) -> None:
    """Handle incoming voice commands.
    
    Args:
        topic: The topic on which the message was received
        payload: Dictionary containing the command data
    """
    try:
        logger.info(f"Processing voice command on Topic: {topic}, Payload: {payload}")

        asyncio.create_task(invoke_conversation(payload["sentences"][-1]["text"], thread_id=1))
    
    except Exception as e:
        logger.error(f"Error processing voice command: {e}", exc_info=True)
