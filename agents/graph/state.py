"""
LangGraph-compatible State schema with message-level IDs.
Tracks per-message steps, RAG/tool/LLM details, and assistant replies.
"""
from typing import Dict, Any, List, Annotated
from langgraph.graph import MessagesState, add_messages
from langchain_core.messages import AnyMessage
import uuid


def gen_id(prefix: str) -> str:
    """Generate a short prefixed UUID."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class Step:
    """Represents one processing step in the flow (RAG, tool_call, llm_response, etc.)"""
    def __init__(self, step_type: str, **kwargs):
        self.id = gen_id("step")
        self.type = step_type
        self.data = kwargs

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "type": self.type, **self.data}


class State(MessagesState):
    """
    LangGraph-compatible conversation state.

    - `messages`: uses LangGraph reducer (add_messages)
    - `metadata`: keyed by user message IDs

    Example structure:
    ```json
    {
      "thread_id": "t-001",
      "messages": [
        {"id": "user-abc123", "role": "user", "content": "What's the weather in Paris?"},
        {"id": "ai-def456", "role": "assistant", "content": "It’s 22°C and clear in Paris."}
      ],
      "metadata": {
        "user-abc123": {
          "steps": [
            {"id": "step-1", "type": "rag", "rag_results": [...]},
            {"id": "step-2", "type": "tool_call", "tool_calls": [...]},
            {"id": "step-3", "type": "llm_response", "output": "..."}
          ],
          "assistant_message": {"id": "ai-def456", "role": "assistant", "content": "..."}
        }
      }
    }
    ```
    """
    messages: Annotated[List[AnyMessage], add_messages]
    thread_id: str
    metadata: Dict[str, Any]

    @classmethod
    def initialize(cls, thread_id: str) -> "State":
        """Initialize a new conversation state."""
        return cls(messages=[], thread_id=thread_id, metadata={})

    def start_turn(self, user_message: Dict[str, Any]):
        """Register a new user message."""
        if "id" not in user_message:
            user_message["id"] = gen_id("user")

        self.metadata[user_message["id"]] = {"steps": [], "assistant_message": None}
        self.messages.append(user_message)
        return user_message["id"]

    def add_step(self, user_message_id: str, step: Step):
        """Add a step linked to a user message."""
        if user_message_id not in self.metadata:
            raise ValueError(f"No metadata found for user message ID: {user_message_id}")

        self.metadata[user_message_id]["steps"].append(step.to_dict())

    def complete_turn(self, user_message_id: str, assistant_message: Dict[str, Any]):
        """Attach the assistant message to a specific user message."""
        if "id" not in assistant_message:
            assistant_message["id"] = gen_id("ai")

        if user_message_id not in self.metadata:
            raise ValueError(f"No metadata found for user message ID: {user_message_id}")

        self.metadata[user_message_id]["assistant_message"] = assistant_message
        self.messages.append(assistant_message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full state."""
        return {
            "thread_id": self.thread_id,
            "messages": self.messages,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "State":
        """Rehydrate from dict."""
        return cls(
            thread_id=data["thread_id"],
            messages=data.get("messages", []),
            metadata=data.get("metadata", {}),
        )
