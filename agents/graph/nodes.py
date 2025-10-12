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
    """Process tool node safely with metadata tracking."""
    messages = state.get("messages", [])
    if not messages:
        print("No messages found in state")
        return state

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if tool_calls:
        print("Tool calls found")
        result = tool_node.invoke(state, config)
        print(result)

        # Identify the user message ID to attach step metadata
        user_message_id = last_message.id
        metadata = state.get("metadata", {})
        message_metadata = metadata.get(user_message_id, {"steps": []})

        # Append step info safely
        message_metadata["steps"].append({
            "id": f"step-tool-{len(message_metadata['steps']) + 1}",
            "type": "tool_call",
            "tool_calls": result
        })

        metadata[user_message_id] = message_metadata

        return {
            "metadata": metadata
        }
    else:
        print("No tool calls found")
        return state


async def memory_state_update(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Rebuild message history from storage and update state."""
    thread_id = config.get("metadata", {}).get("thread_id")
    thread_history: list[dict[str, str]] = await get_messages(thread_id)
    messages = state.get("messages", [])
    last_human_message = messages[-1] if messages else None

    existing_messages = []

    if thread_history:
        for msg_pair in thread_history:
            user_msg = msg_pair.get("user_message")
            ai_msg = msg_pair.get("ai_message")

            if user_msg:
                existing_messages.append(HumanMessage(content=user_msg))
            if ai_msg:
                existing_messages.append(AIMessage(content=ai_msg))

    # Rebuild message list: clear current messages, add old ones, then the latest human message
    all_messages = [RemoveMessage(id=m.id) for m in messages] + existing_messages
    if last_human_message:
        all_messages.append(HumanMessage(content=last_human_message.content))

    # Preserve metadata structure
    metadata = state.get("metadata", {})
    last_user_id = last_human_message.id if last_human_message and hasattr(last_human_message, "id") else None
    if last_user_id and last_user_id not in metadata:
        metadata[last_user_id] = {"steps": [], "assistant_message": None}

    return {
        "messages": all_messages,
        "metadata": metadata,
        "thread_id": thread_id,
    }


async def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process conversation through the language model with context."""
    print(f"state: {state}")

    messages = state.get("messages", [])
    metadata = state.get("metadata", {})
    summary = state.get("summary", "")

    current_message = None
    conversation_history = []

    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage):
            current_message = last_msg.content
            conversation_history = messages[:-1]
        else:
            conversation_history = messages

    recent_history = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history

    # Build system context
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

    if summary and summary.strip():
        context_parts.append(
            f"""=== CONVERSATION SUMMARY CONTEXT ===
            {summary}
            === END SUMMARY ==="""
        )

    combined_context = "\n\n".join(context_parts)
    prompt_parts = [{"role": "system", "content": combined_context}]

    # Build conversation prompt
    for msg in recent_history or []:
        if isinstance(msg, HumanMessage):
            prompt_parts.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            prompt_parts.append({"role": "assistant", "content": msg.content})

            msg_id = msg.id
            msg_meta = metadata.get(msg_id, {}) if msg_id else {}
            steps = msg_meta.get("steps", [])

            for step in steps or []:
                step_type = step.type
                if step_type == "rag":
                    rag_results = step.rag_results
                    prompt_parts.append({
                        "role": "system",
                        "content": f"RAG Results: {rag_results}"
                    })
                elif step_type == "tool_call":
                    tool_calls = step.tool_calls
                    prompt_parts.append({
                        "role": "system",
                        "content": f"Tool Calls: {tool_calls}"
                    })
                elif step_type == "llm_response":
                    output = step.output
                    if output:
                        prompt_parts.append({
                            "role": "assistant",
                            "content": output
                        })

    if current_message:
        prompt_parts.append({"role": "user", "content": current_message})

    chat_prompt = ChatPromptTemplate.from_messages(prompt_parts)
    formatted_messages = await chat_prompt.aformat_messages()

    # Debug info
    logger.debug("=== STATE DEBUG ===")
    logger.debug(f"Messages count: {len(messages)}")
    logger.debug(f"Recent history count: {len(recent_history)}")
    logger.debug(f"Metadata keys: {list(metadata.keys())}")
    logger.debug("===================")

    response = await llm_chat_model_with_tools.ainvoke(formatted_messages, config=config)
    print(f"LLM response: {response}")

    return {"messages": [response]}



async def generate_embeddings_for_query(message: str) -> List[float]:
    """Generate embeddings for the given query text."""
    return await embeddings_generator.aembed_query(message)


def path_selector_post_llm_call(state: State) -> str:
    """Decide whether to summarize conversation or end."""
    messages = state.get("messages", [])
    if not messages:
        return "workflow_completion"

    last_message = messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "tools_execution"

    return "summarize_conversation" if len(messages) > 2 else "workflow_completion"


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
    results = state.get("metadata", {}).get("doc_rag_results", [])
    retrieved_docs = '\n'.join(item.get("data", "") for item in results)
    return {"doc_rag_results": retrieved_docs}


def retrieve_data_from_web_RAG(state: State) -> Dict[str, Any]:
    """Placeholder for web RAG retrieval."""
    retrieved_results = state.get("metadata", {}).get("web_rag_results", [])
    return {"web_rag_results": retrieved_results}


async def summarize_conversation(state: State) -> Dict[str, Any]:
    """Summarize the conversation into a compact text."""
    summary = state.get("summary", "")
    messages = state.get("messages", [])

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

    chat_messages = [{"role": "system", "content": summary_message}] + [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": getattr(m, "content", "")}
        for m in messages
    ]

    response = await llm_chat_model.ainvoke(
        chat_messages,
        config={
            "metadata": {
                "thread_id": state.get("thread_id", ""),
                "source_application": "summarize_conversation"
            }
        }
    )

    delete_messages = [
        RemoveMessage(id=getattr(m, "id", "")) for m in messages[:-2]
    ] if len(messages) > 2 else []

    return {"summary": getattr(response, "content", ""), "messages": delete_messages}


async def workflow_completion(state: State) -> Dict[str, Any]:
    """Workflow completion node."""
    return {"messages": state.get("messages", [])}
