#!/usr/bin/env python3
"""Conversation bot with streaming + final payload."""
import asyncio
from typing import Dict, Any, AsyncGenerator, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
from .graph import build_workflow
from .storage import delete_messages, save_message
from .text_writer import write_response_to_bus

# Load environment and workflow
load_dotenv()
workflow = build_workflow()


async def run_conversation(
    message: str,
    thread_id: int = 1,
) -> AsyncGenerator[Dict[str, Union[str, bool]], None]:
    """
    Stream workflow results.
    Yields dict payloads:
        {"status": "stream", "chunk": "..."}   -> during streaming
        {"status": "complete", "response": "..."} -> once finished
    """
    buffer = ""
    max_buffer_size = 50
    full_response = ""

    async for stream_mode, chunk in workflow.astream(
        {"messages": [HumanMessage(content=message)]},
        {"configurable": {"thread_id": thread_id}},
        stream_mode=["messages", "updates"],
    ):

        # when stream_mode is messages, it means we are streaming the response
        if stream_mode == "messages":
            for msg in chunk:
                if isinstance(msg, AIMessageChunk):
                    buffer += msg.content
                    full_response += msg.content

                    # Stream buffer if it exceeds max size
                    if len(buffer) >= max_buffer_size:
                        response = {"status": "stream", "node": "conversation", "response": buffer}
                        yield response
                        buffer = ""

        # when stream_mode is updates, it means we are yielding the updates from summarize_conversation and responding the summary to the user
        elif stream_mode == "updates":
            if chunk.get("summarize_conversation"):
                if chunk.get("summarize_conversation").get("summary"):
                    yield {
                        "status": "update", 
                        "node": "summarize_conversation", 
                        "response": chunk.get("summarize_conversation").get("summary")
                    }

    # Stream any remaining buffer
    if buffer:
        response = {"status": "stream", "node": "conversation", "response": buffer}
        yield response

    # Final payload, stream the full response
    response = {
        "status": "complete", 
        "node": "conversation", 
        "response": full_response
    }
    yield response


async def invoke_conversation(
    user_message: str,
    thread_id: int = 1,
) -> str:
    """
    Orchestrator function for external callers (API, CLI, etc.).

    - Invokes the conversation graph via run_conversation
    - Processes streaming responses (e.g., TTS or live feedback)
    - Handles updates like summarization
    - On completion, persists the conversation into DB
    - Returns the final AI response as a string
    """
    ai_message: str = ""
    summary: str = ""

    async for payload in run_conversation(user_message, thread_id):

        node = payload["node"]
        status = payload["status"]
        
        # Handle conversation streaming chunks
        if node == "conversation":
            if status == "stream":
                # send chunk to TTS system
                payload = {
                    "thread_id": thread_id,
                    "llm_response": payload["response"]
                }
                await write_response_to_bus(payload)

            elif status == "complete":
                ai_message = payload["response"]

        # Handle summarization or other updates
        elif node == "summarize_conversation":
            if status == "update":
                summary = payload["response"]

    if node == "conversation" and status == "complete":
        # Persist conversation after completion
        await save_message(thread_id, user_message, ai_message, summary)

    return ai_message

async def main():
    """Example CLI runner for testing the orchestrator."""
    thread_id = 11
    await delete_messages(thread_id)

    messages = [
        "Hello, how are you?",
        "Which model are you using?",
        "Do you know my name?",
    ]

    for user_message in messages:
        print(f"\nUser: {user_message}")
        response = await invoke_conversation(user_message, thread_id)
        print(f"AI: {response}")


if __name__ == "__main__":
    asyncio.run(main())
