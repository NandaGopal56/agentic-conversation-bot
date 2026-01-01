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

def on_voice_command(topic: str, message: str) -> None:
    """Handle incoming voice commands.
    
    Args:
        topic: The topic on which the message was received
        message: The text of the voice command
    """
    try:
        logger.info(f"Processing voice command on Topic: {topic}, message: {message}")

        asyncio.create_task(invoke_conversation(message, thread_id=1))
    
    except Exception as e:
        logger.error(f"Error processing voice command: {e}", exc_info=True)
