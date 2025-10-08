"""
Node definitions for the conversation graph.
Each function represents a node in the graph.
"""
import logging
from typing import Dict, Any, List, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.types import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from ..logger import logger
from ..storage import get_messages, save_message, delete_messages
from .state import State
from ..tools.basic_tools import basic_tools

load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

# define the final list of tools
tools = basic_tools
tool_node = ToolNode(tools=tools)

llm_chat_model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
llm_chat_model_with_tools = llm_chat_model.bind_tools(tools)
embeddings_generator = OpenAIEmbeddings(model="text-embedding-ada-002")


def tool_node_processor(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process tool node."""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        print("Tool calls found")
        result = tool_node.invoke(state, config)
        print(result)
        return {"tool_calls": result}
    else:
        print("No tool calls found")
        return state

async def memory_state_update(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Rebuild message history from storage and update state."""
    thread_id = config['metadata']['thread_id']
    thread_history: List[Dict[str, str]] = await get_messages(thread_id)
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
        "messages": all_messages,
        "thread_id": thread_id
    }

async def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process conversation through the language model with context."""
    print(f'state: {state}')
    messages = state.get("messages", [])
    current_message = None
    conversation_history = []

    if messages:
        current_message = messages[-1].content if messages[-1].type == "human" else None
        conversation_history = messages[:-1] if current_message else messages

    # Recent history: last 2 exchanges
    recent_history = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history

    context_parts = [
        """You are a helpful AI assistant designed to behave like a voice assistant (e.g., Alexa or Google Assistant).
        Always answer clearly, concisely, and in a natural conversational style.
        Prioritize providing direct, useful information without unnecessary elaboration.
        If you don’t know the answer, say so plainly instead of inventing information.
        Follow user instructions carefully and avoid going off-topic.
        For multi-step or complex tasks, break responses into simple, actionable steps.
        Stay polite, neutral, and professional at all times.
        Do not include opinions, speculation, or personal experiences.
        Do not generate disallowed, unsafe, or harmful content.
        Avoid overly long answers unless explicitly requested.
        When clarification is needed, ask a short and direct follow-up question."""
    ]

    if state.get("summary") and state["summary"].strip():
        context_parts.append(
            f"""=== CONVERSATION SUMMARY CONTEXT ===
            The following is a summary of earlier parts of this conversation:
            {state['summary']}
            === END CONVERSATION SUMMARY ===""")

    if state.get("doc_rag_results") and state["doc_rag_results"].strip():
        context_parts.append(
            f"""=== DOCUMENT KNOWLEDGE CONTEXT ===
            The following relevant information has been retrieved from documents:
            {state['doc_rag_results']}
            === END DOCUMENT CONTEXT ===""")

    if state.get("web_rag_results") and state["web_rag_results"].strip():
        context_parts.append(
            f"""=== WEB SEARCH CONTEXT ===
            The following relevant information has been retrieved from web search:
            {state['web_rag_results']}
            === END WEB SEARCH CONTEXT ===""")


    # Combine all context into ONE system message at the top
    combined_context = "\n\n".join(context_parts)
    
    # Build prompt
    prompt_parts = [{"role": "system", "content": combined_context}]

    
    for msg in recent_history:
        if msg.type == "human":
            prompt_parts.append(msg)
        if msg.type == "ai":
            prompt_parts.append(msg)
            if msg.tool_calls:
                prompt_parts.append(state['tool_calls']['messages'][0])
                prompt_parts.append(state['tool_calls']['messages'][1])

    if current_message:
        prompt_parts.append({"role": "user", "content": current_message})

    chat_prompt = ChatPromptTemplate.from_messages(prompt_parts)
    formatted_messages = await chat_prompt.aformat_messages()

    # Debug logging
    logger.debug("=== STATE DEBUG ===")
    logger.debug(f"Summary exists: {bool(state.get('summary'))}")
    logger.debug(f"Doc RAG exists: {bool(state.get('doc_rag_results'))}")
    logger.debug(f"Web RAG exists: {bool(state.get('web_rag_results'))}")
    logger.debug(f"Total messages count: {len(messages)}")
    logger.debug(f"Current message extracted: {bool(current_message)}")
    logger.debug(f"History messages count: {len(conversation_history)}")
    logger.debug(f"Recent history count: {len(recent_history)}")
    logger.debug(f"Total prompt parts: {len(prompt_parts)}")
    logger.debug("===================")

    # Call LLM
    response = await llm_chat_model_with_tools.ainvoke(formatted_messages, config=config)
    print(f'LLM response: {response}')
    return {"messages": [response]}

async def generate_embeddings_for_query(message: str) -> List[float]:
    """Generate embeddings for the given query text."""
    return await embeddings_generator.aembed_query(message)

def path_selector_post_llm_call(state: State) -> str:
    """Decide whether to summarize conversation or end."""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools_execution"

    return "summarize_conversation" if len(state["messages"]) > 2 else "workflow_completion"


def branch_selection_for_RAG(state: State) -> Union[str, List[str]]:
    """Decide which RAG branches to run based on state."""
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
    """Placeholder for document RAG retrieval."""
    results = []  # TODO: implement actual document retrieval
    retrieved_docs = '\n'.join(item['data'] for item in results)
    return {"doc_rag_results": retrieved_docs}

def retrieve_data_from_web_RAG(state: State) -> Dict[str, Any]:
    """Placeholder for web RAG retrieval."""
    retrieved_results = []  # TODO: implement actual web search
    return {"web_rag_results": retrieved_results}

async def summarize_conversation(state: State) -> Dict[str, Any]:
    """Summarize the conversation into a compact text."""
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

    messages = [{"role": "system", "content": summary_message}] + [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in state["messages"]
    ]
    
    response = await llm_chat_model.ainvoke(messages, config={
        "metadata": {
            "thread_id": state.get("thread_id"),
            "source_application": "summarize_conversation"
        }
    })

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


async def workflow_completion(state: State) -> Dict[str, Any]:
    """Workflow completion node."""
    return {"messages": state["messages"]}