"""
Workflow definition for the conversation graph.
Builds and configures the state graph with all nodes and edges.
"""
from typing import Dict, Any
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver as CheckpointMemorySaver
from .state import State

from .nodes import (
    memory_state_update,
    call_model,
    branch_selection_for_RAG,
    retrieve_data_from_doc_RAG,
    retrieve_data_from_web_RAG,
    should_continue,
    summarize_conversation,
    workflow_completion
)

def build_workflow() -> StateGraph:
    """Build and configure the conversation workflow graph.
    
    Returns:
        StateGraph: Configured state graph with all nodes and edges.
    """
    # Initialize the workflow with our State class
    workflow = StateGraph(State)
    
    # Add all nodes to the workflow
    workflow.add_node("memory_state_update", memory_state_update)
    workflow.add_node("doc_rag_search", retrieve_data_from_doc_RAG)
    workflow.add_node("web_rag_search", retrieve_data_from_web_RAG)
    workflow.add_node("conversation", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_node("workflow_completion", workflow_completion)
    
    # Define the workflow edges
    workflow.add_edge(START, "memory_state_update")
    workflow.add_conditional_edges(
        source="memory_state_update",
        path=branch_selection_for_RAG,
        path_map={
            "doc_rag_search": "doc_rag_search",
            "web_rag_search": "web_rag_search",
            "conversation": "conversation"
        }
    )
    
    # Connect RAG nodes to conversation
    workflow.add_edge("doc_rag_search", "conversation")
    workflow.add_edge("web_rag_search", "conversation")
    
    # Add conditional edges after conversation
    workflow.add_conditional_edges(
        source="conversation",
        path=should_continue,
        path_map={
            "summarize_conversation": "summarize_conversation",
            "workflow_completion": "workflow_completion"
        }
    )
    
    # Connect the rest of the workflow
    workflow.add_edge("summarize_conversation", "workflow_completion")
    workflow.add_edge("workflow_completion", END)
    
    # Compile the workflow with memory checkpointing
    return workflow.compile(checkpointer=CheckpointMemorySaver())