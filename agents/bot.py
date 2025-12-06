import asyncio
from typing import Dict, AsyncGenerator, Union
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from .graph import build_workflow
from .storage import add_message, add_tool_call, update_tool_result

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


class NodeCompletionProcessor:
    """Processes 'values' stream mode - handles actions after node completion."""
    
    def __init__(self, thread_id: int, user_message: str):
        self.thread_id = thread_id
        self.user_message = user_message
        self.last_user_message_id = None
        self.last_ai_message_id = None
        self.final_ai_response = None

    async def process_node_completion(self, state_snapshot) -> None:
        """
        Process complete state after a node finishes execution.
        Detect node name and call storage APIs identical to first code logic.
        """
        node_name = state_snapshot.get("langgraph_node")     # <---- extract node name
        messages = state_snapshot.get("messages", [])

        print(f'Snapshot: {state_snapshot}, Node: {node_name}')

        last_message = messages[-1]

        if isinstance(last_message, HumanMessage) and node_name == "memory_state_update":
            # self.last_user_message_id = await add_message(
            #     thread_id=self.thread_id,
            #     role="user",
            #     content=last_message.content
            # )
            return

        if isinstance(last_message, AIMessage) and node_name == "call_model":
            
            # Save AI message
            # self.last_ai_message_id = await add_message(
            #     thread_id=self.thread_id,
            #     role="assistant",
            #     content=last_message.content
            # )

            self.final_ai_response = last_message.content

            # Save tool calls if any
            if getattr(last_message, "tool_calls", None):
                # await add_tool_call(
                #     message_id=self.last_ai_message_id,
                #     tool_args=last_message.tool_calls,
                #     tool_id=last_message.tool_calls[0].id
                # )
                pass

            print("[NodeCompletionProcessor] call_model completed.")
            return

        if isinstance(last_message, ToolMessage) and node_name == "tool_node_processor":
            # await update_tool_result(
            #     message_id=self.last_ai_message_id,
            #     tool_result=last_message.model_dump()
            # )
            print("[NodeCompletionProcessor] tool execution stored.")
            return

        if node_name == "summarize_conversation":
            return

    async def finalize(self) -> None:
        """Final actions after entire graph execution completes."""
        if self.final_ai_response:
            print(f"[NodeCompletionProcessor] Conversation finalized for thread {self.thread_id}")



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
    completion_processor = NodeCompletionProcessor(thread_id, message)

    # Stream through workflow - delegate everything to processors
    async for stream_mode, stream_data in workflow.astream(
        {"messages": [HumanMessage(content=message)]},
        {"configurable": {"thread_id": thread_id}},
        stream_mode=["messages", "custom"],
    ):
        if stream_mode == "messages":
            async for payload in token_processor.process_chunk(stream_data):
                yield payload
        
        # elif stream_mode == "custom":
        #     await completion_processor.process_node_completion(stream_data)
    
    # Finalize both processors
    async for payload in token_processor.finalize():
        yield payload
    
    # await completion_processor.finalize()


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
    thread_id = 12
    
    # Demo conversation scenarios
    demo_conversations = [
        # "Hello! How are you today?",
        # "What's the weather like in New York and distance between New York and Paris?",
        # "Can you help me find restaurants near MG Road, Bangalore?",
        "Tell me a short joke about programming",
        "What's 25 * 47?",
    ]
    
    for user_message in demo_conversations:
        print(f"\n{'='*60}")
        print(f"User: {user_message}", end='\n\n')
        response = await invoke_conversation(user_message, thread_id)
        print(f"Final AI Response: {response}", end='\n\n')
        print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())