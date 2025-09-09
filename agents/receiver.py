"""
Agents module for the Agetic Conversation Bot.

This module contains the core agent implementations that process voice commands
and manage conversations.
"""
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def on_voice_command(topic: str, payload: Dict[str, Any]) -> None:
    """Handle incoming voice commands.
    
    Args:
        topic: The topic on which the message was received
        payload: Dictionary containing the command data
    """
    try:
        logger.info(f"Processing voice command on {topic}")
        # Process the command here
        logger.debug(f"Command data: {payload}")
        
        # Example processing
        if 'text' in payload:
            logger.info(f"Command text: {payload['text']}")
            
    except Exception as e:
        logger.error(f"Error processing voice command: {e}", exc_info=True)


