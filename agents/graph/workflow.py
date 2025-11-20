"""
Workflow definition for the conversation graph.
Builds and configures the state graph with all nodes and edges.
"""
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver as CheckpointMemorySaver
from .state import State

from .nodes import (
    memory_state_update,
    call_model,
    branch_selection_for_RAG,
    retrieve_data_from_doc_RAG,
    retrieve_data_from_web_RAG,
    summarize_conversation,
    workflow_completion,
    tool_node_processor,
    path_selector_post_llm_call
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
    workflow.add_node("call_model", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_node("workflow_completion", workflow_completion)
    workflow.add_node("tools_execution", tool_node_processor)
    
    # Define the workflow edges
    workflow.add_edge(START, "memory_state_update")
    workflow.add_conditional_edges(
        source="memory_state_update",
        path=branch_selection_for_RAG,
        path_map={
            "doc_rag_search": "doc_rag_search",
            "web_rag_search": "web_rag_search",
            "call_model": "call_model"
        }
    )
    
    # Connect RAG nodes to call_model
    workflow.add_edge("doc_rag_search", "call_model")
    workflow.add_edge("web_rag_search", "call_model")
    
    # Add conditional edges after call_model
    workflow.add_conditional_edges(
        source="call_model",
        path=path_selector_post_llm_call,
        path_map={
            "tools_execution": "tools_execution",
            "summarize_conversation": "summarize_conversation",
            "workflow_completion": "workflow_completion"
        }
    )

    workflow.add_edge("tools_execution", "call_model")

    # Connect the rest of the workflow
    workflow.add_edge("summarize_conversation", "workflow_completion")
    workflow.add_edge("workflow_completion", END)
    
    # Compile the workflow with memory checkpointing
    return workflow.compile(checkpointer=CheckpointMemorySaver())