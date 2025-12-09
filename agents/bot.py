import asyncio
import sys
from typing import Dict, AsyncGenerator, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from .graph import build_workflow

# Load environment and build workflow
load_dotenv()
workflow = build_workflow()


class TokenStreamProcessor:
    """Processes 'messages' stream mode - handles real-time LLM token streaming."""
    
    def __init__(self, max_buffer_size: int = 50):
        self.buffer = ""
        self.max_buffer_size = max_buffer_size
    
    async def process_chunk(self, chunk) -> AsyncGenerator[Dict[str, str], None]:
        """
        Process each streaming LLM token chunk as it arrives.
        Buffers tokens and yields when punctuation is hit or buffer is full.
        """

        langgraph_node = chunk[1].get("langgraph_node")

        # final AI response should come only from call_model node. Rest all nodes are internal nodes.
        if langgraph_node != "call_model":
            return

        for msg_tuple in chunk:
            # Extract token from tuple or direct message
            token = msg_tuple[0] if isinstance(msg_tuple, tuple) else msg_tuple
            
            if hasattr(token, 'content') and token.content:
                self.buffer += token.content
                
                # Flush buffer on punctuation or size limit
                should_flush = (
                    any(p in self.buffer for p in [".", "!", "?", ","]) or 
                    len(self.buffer) >= self.max_buffer_size
                )
                
                if should_flush:
                    yield {
                        "status": "streaming",
                        "node": "call_model",
                        "response": self.buffer.strip()
                    }
                    self.buffer = ""
    
    async def finalize(self) -> AsyncGenerator[Dict[str, str], None]:
        """Flush any remaining buffered tokens and signal completion."""
        if self.buffer.strip():
            yield {
                "status": "streaming",
                "node": "call_model",
                "response": self.buffer.strip()
            }
            self.buffer = ""


async def run_conversation(
    message: str,
    thread_id: int = 1,
) -> AsyncGenerator[Dict[str, Union[str, bool]], None]:
    """
    Stream workflow results using both 'messages' and 'values' stream modes.
    Delegates all processing to specialized processors - takes no direct action.
    
    - messages mode: Real-time token streaming (via TokenStreamProcessor)
    - values mode: Node completion handling (via NodeCompletionProcessor)
    
    Args:
        message: User input message
        thread_id: Conversation thread identifier
    
    Yields:
        Dict payloads from both stream processors
    """
    # Initialize processors
    token_processor = TokenStreamProcessor()

    # Stream through workflow - delegate everything to processors
    async for stream_mode, stream_data in workflow.astream(
        {"messages": [HumanMessage(content=message)]},
        {"configurable": {"thread_id": thread_id}},
        stream_mode=["messages"],
    ):
        if stream_mode == "messages":
            async for payload in token_processor.process_chunk(stream_data):
                yield payload
        
    # Finalize both processors
    async for payload in token_processor.finalize():
        yield payload


async def invoke_conversation(
    user_message: str, 
    thread_id: int = 1
) -> str:
    """
    Orchestrator function for external callers (API, CLI, etc.).
    Processes streaming responses and returns the final AI response.
    """
    final_ai_response = ""

    async for payload in run_conversation(user_message, thread_id):
        node = payload["node"]
        status = payload["status"]
        response = payload["response"]
        
        # print(f"Node: {node}, Status: {status}, Response: {response}")

        # Handle real-time LLM token streaming
        if node == "call_model" and status == "streaming":
            # Send chunk to TTS system for real-time speech
            # await write_response_to_bus({
            #     "thread_id": thread_id,
            #     "llm_response": response
            # })
            pass
        
        # Accumulate streaming response to return final result
        if status == "streaming":
            final_ai_response += response
        
    return final_ai_response


async def main():
    """Example CLI runner with demo conversations."""
    thread_id = 13
    mode = None
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip()
    else:
        mode = "2"
    
    if mode == "1":
        while True:
            try:
                user_message = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_message:
                continue
            if user_message.lower() in {"exit", "quit"}:
                break
            response = await invoke_conversation(user_message, thread_id)
            print(f"Assistant: {response}")
    else:
        demo_conversations = [
            # "Hello! How are you today?",
            # "What's the weather like in New York and distance between New York and Paris?",
            # "Can you help me find restaurants near MG Road, Bangalore?",
            # "Tell me a short joke about programming",
            # "What's 25 * 47?",
            "can you tell me whats you are seeing now from camera feed ?"
        ]
        
        for user_message in demo_conversations:
            print(f"\n{'='*60}")
            print(f"User: {user_message}", end='\n\n')
            response = await invoke_conversation(user_message, thread_id)
            print(f"Final AI Response: {response}", end='\n\n')
            print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())