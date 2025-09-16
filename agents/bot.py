import langchain
import os
import logging
import asyncio

from typing import Dict, Any, List, Union
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver as CheckpointMemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from logger import logger

from storage import get_messages, save_message, delete_messages

load_dotenv()



# ------------------------------
# State Definition
# ------------------------------
class State(MessagesState):
    """Extended state class"""
    summary: str
    thread_id: int
    doc_rag_results: str = 'Document snippet: Gopal is a common Indian name.'
    web_rag_results: str = 'Web search snippet: Gopal is a common Indian name.'


# ------------------------------
# Model Initialization
# ------------------------------
llm_chat_model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
embeddings_generator = OpenAIEmbeddings(model="text-embedding-ada-002")
memory_saver = CheckpointMemorySaver()


# ------------------------------
# Nodes
# ------------------------------
async def memory_state_update(state: State) -> Dict[str, List[AIMessage]]:
    """Rebuild message history from storage and update state."""

    thread_history: List[Dict[str, str]] = await get_messages(state.get("thread_id"))
    last_human_message = state.get("messages")[-1]

    existing_messages = []
    summary = ""

    if thread_history:
        for msg_pair in thread_history:
            if msg_pair["user_message"]:
                existing_messages.append(HumanMessage(content=msg_pair["user_message"]))
            if msg_pair["ai_message"]:
                existing_messages.append(AIMessage(content=msg_pair["ai_message"]))
            if msg_pair["summary"]:
                summary = msg_pair["summary"]

    # Replace in-memory messages with DB messages + current human input
    all_messages = [RemoveMessage(id=m.id) for m in state["messages"]] \
        + existing_messages \
        + [HumanMessage(content=last_human_message.content)]

    return {
        "summary": summary,
        "messages": all_messages
    }



async def call_model(state: dict) -> dict:
    """Build structured prompt with correct order and all context pieces."""
    logger.debug(state)

    messages = state.get("messages", [])
    current_message = None
    conversation_history = []

    if messages:
        # Last human message is the current message
        current_message = messages[-1].content if messages[-1].type == "human" else None
        # All previous messages are conversation history
        conversation_history = messages[:-1] if current_message else messages

    # Recent history: last 2 exchanges
    recent_history = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history

    # ----------------
    # Build system context
    # ----------------
    context_parts = []

    base_instruction = (
        "You are a helpful AI assistant. Answer all questions to the best of your ability. "
        "Use the provided context information when relevant to answer the user's question. "
        "If you are not aware of any information, explicitly say so instead of making things up. "
        "Maintain a conversational and helpful tone throughout your responses."
    )
    context_parts.append(base_instruction)

    if state.get("summary") and state["summary"].strip():
        context_parts.append(
            "=== CONVERSATION SUMMARY CONTEXT ===\n"
            "The following is a summary of earlier parts of this conversation:\n"
            f"{state['summary']}\n"
            "=== END CONVERSATION SUMMARY ==="
        )

    if state.get("doc_rag_results") and state["doc_rag_results"].strip():
        context_parts.append(
            "=== DOCUMENT KNOWLEDGE CONTEXT ===\n"
            "The following relevant information has been retrieved from documents:\n"
            f"{state['doc_rag_results']}\n"
            "=== END DOCUMENT CONTEXT ==="
        )

    if state.get("web_rag_results") and state["web_rag_results"].strip():
        context_parts.append(
            "=== WEB SEARCH CONTEXT ===\n"
            "The following relevant information has been retrieved from web search:\n"
            f"{state['web_rag_results']}\n"
            "=== END WEB SEARCH CONTEXT ==="
        )

    if recent_history:
        history_text = "=== RECENT CONVERSATION CONTEXT ===\n"
        history_text += "The following shows the last 2 messages from your recent conversation:\n"
        for msg in recent_history:
            role = "User" if msg.type == "human" else "Assistant"
            history_text += f"[{role}] {msg.content}\n"
        history_text += "=== END RECENT CONVERSATION CONTEXT ==="
        context_parts.append(history_text)

    # Combine all context into ONE system message at the top
    combined_context = "\n\n".join(context_parts)

    # ----------------
    # Build prompt
    # ----------------
    prompt_parts = [("system", combined_context)]

    if current_message:
        prompt_parts.append(("user", current_message))

    chat_prompt = ChatPromptTemplate.from_messages(prompt_parts)
    formatted_messages = await chat_prompt.aformat_messages()

    # Debug
    logger.debug("=== STATE DEBUG ===")
    logger.debug(f"Summary exists: {bool(state.get('summary'))}")
    logger.debug(f"Doc RAG exists: {bool(state.get('doc_rag_results'))}")
    logger.debug(f"Web RAG exists: {bool(state.get('web_rag_results'))}")
    logger.debug(f"Total messages count: {len(messages)}")
    logger.debug(f"Current message extracted: {bool(current_message)}")
    logger.debug(f"History messages count: {len(conversation_history)}")
    logger.debug(f"Recent history count: {len(recent_history)}")
    logger.debug(f"Total prompt parts: {len(prompt_parts)}")
    logger.debug(f"Prompt: {formatted_messages}")
    logger.debug("===================")

    # Call LLM
    response = await llm_chat_model.ainvoke(formatted_messages)

    return {"messages": [response]}


async def generate_embeddings_for_query(message: str) -> List[float]:
    return await embeddings_generator.aembed_query(message)


def should_continue(state: State) -> str:
    """Decide whether to summarize conversation or end."""
    return "summarize_conversation" if len(state["messages"]) > 2 else "save_state"


def branch_selection_for_RAG(state: State) -> Union[str, List[str]]:
    """Decide which RAG branches to run."""
    doc_enabled = state.get("is_doc_rag_enabled", False)
    web_enabled = state.get("is_search_enabled", False)

    if doc_enabled and web_enabled:
        return ["doc_rag_search", "web_rag_search"]
    if doc_enabled:
        return "doc_rag_search"
    if web_enabled:
        return "web_rag_search"
    return "conversation"


def retrieve_data_from_doc_RAG(state: State) -> Dict[str, Any]:
    """Placeholder for doc RAG retrieval."""
    results = []  # TODO: implement
    retrieved_docs = '\n'.join(item['data'] for item in results)
    return {"doc_rag_results": retrieved_docs}


def retrieve_data_from_web_RAG(state: State) -> Dict[str, Any]:
    """Placeholder for web RAG retrieval."""
    retrieved_results = []  # TODO: implement
    return {"web_rag_results": retrieved_results}


async def summarize_conversation(state: State) -> Dict[str, Any]:
    """Summarize conversation into compact text."""
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is the summary of the conversation so far: {summary}\n\n"
            "Extend the summary to include the new messages."
            "Do not end or add any questions or open-ended prompts."
            "Be concise and do not add any additional information."
            "End only with the summary and no additional text."
        )
    else:
        summary_message = "Create a summary of the conversation below."

    messages = [SystemMessage(content=summary_message)] + state["messages"]
    response = await llm_chat_model.ainvoke(messages, config={
        "metadata": {
            "thread_id": state.get("thread_id"),
            "source_application": "summarize_conversation"
        }
    })

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    return {"summary": response.content, "messages": delete_messages}


async def save_state(state: State):
    await save_message(
        state.get("thread_id"),
        state["messages"][-2].content,
        state["messages"][-1].content,
        state.get("summary", "")
    )

    return state

# ------------------------------
# Workflow Builder
# ------------------------------
def build() -> StateGraph:
    workflow = StateGraph(State)

    workflow.add_node("memory_state_update", memory_state_update)
    workflow.add_node("doc_rag_search", retrieve_data_from_doc_RAG)
    workflow.add_node("web_rag_search", retrieve_data_from_web_RAG)
    workflow.add_node("conversation", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_node("save_state", save_state)

    workflow.add_edge(START, "memory_state_update")
    workflow.add_conditional_edges("memory_state_update", branch_selection_for_RAG)
    workflow.add_edge("doc_rag_search", "conversation")
    workflow.add_edge("web_rag_search", "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", "save_state")
    workflow.add_edge("save_state", END)

    return workflow.compile(checkpointer=memory_saver)


# ------------------------------
# Streaming Debug Utility
# ------------------------------
def pretty_print_stream_chunk(chunk):
    for node, updates in chunk.items():
        if "messages" in updates and node in ('memory_state_update', 'conversation'):
            updates["messages"][-1].pretty_print()
        else:
            logger.debug("updates:", updates)
        logger.debug("\n")


async def main():
    thread_id = 1

    await delete_messages(thread_id)

    graph = build()

    messages = [
        "Hello",
        "My name is Gopal",
        "What is my name?",
        "I like hiking on weekends and reading books",
        "What are my hobbies?"
    ]

    for message in messages:
        async for chunk in graph.astream(
            {"messages": [message], "thread_id": thread_id},
            config={"configurable": {"thread_id": thread_id}}
        ):
            pretty_print_stream_chunk(chunk)


# ------------------------------
# Example Run
# ------------------------------
if __name__ == "__main__":

    asyncio.run(main())
