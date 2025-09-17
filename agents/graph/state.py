"""
State management for the conversation graph.
Defines the State class and related state management utilities.
"""
from typing import Dict, Any, List, Union, Optional
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage

class State(MessagesState):
    """Extended state class for conversation management."""
    summary: str = ""
    thread_id: int = 1
    doc_rag_results: str = ''
    web_rag_results: str = ''
    is_doc_rag_enabled: bool = False
    is_search_enabled: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'State':
        """Create a State instance from a dictionary."""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert State to dictionary."""
        return {
            "summary": self.summary,
            "thread_id": self.thread_id,
            "doc_rag_results": self.doc_rag_results,
            "web_rag_results": self.web_rag_results,
            "is_doc_rag_enabled": self.is_doc_rag_enabled,
            "is_search_enabled": self.is_search_enabled,
            "messages": getattr(self, "messages", [])
        }
