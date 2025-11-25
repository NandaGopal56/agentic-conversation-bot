"""
Node definitions for the conversation graph.
Each function represents a node in the graph.
"""
import logging
from typing import Dict, Any, List, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage, ToolMessage, SystemMessage
from langchain_core.messages.tool import tool_call
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.types import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langgraph.config import get_stream_writer
from ..logger import logger
from .state import State
from ..tools.basic_tools import basic_tools
from ..storage import update_tool_result, add_tool_call, add_message

load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

# define the final list of tools
tools = basic_tools
tool_node = ToolNode(tools=tools)

llm_chat_model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
llm_chat_model_with_tools = llm_chat_model.bind_tools(tools)
embeddings_generator = OpenAIEmbeddings(model="text-embedding-ada-002")



async def tool_node_processor(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process tool node safely with metadata tracking."""
    messages = state.get("messages", [])
    if not messages:
        # print("No messages found in state")
        return state

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if tool_calls:
        result = tool_node.invoke(tool_calls, config)

        tool_messages = result.get("messages", [])

        for tool_message in tool_messages:
            await update_tool_result(
                message_id=state.get("last_ai_message_id"),
                call_id=tool_message.tool_call_id,
                output_json=tool_message.content
            )
        
        return {
            "messages": tool_messages
        }
    else:
        return state


async def memory_state_update(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Rebuild message history from storage and update state."""

    human_message_id = await add_message(
        thread_id=config.get("configurable", {}).get("thread_id"), 
        role="user", 
        content=state.get("messages", [])[-1].content
    )

    # print("Memory state update")
    # print('State at memory state update: ', state)
    # print('-' * 50)

    # thread_id = config.get("metadata", {}).get("thread_id")
    # thread_history: list[dict[str, str]] = await get_messages(thread_id)
    # messages = state.get("messages", [])
    # last_human_message = messages[-1] if messages else None

    # existing_messages = []

    # if thread_history:
    #     for msg_pair in thread_history:
    #         user_msg = msg_pair.get("user_message")
    #         ai_msg = msg_pair.get("ai_message")

    #         if user_msg:
    #             existing_messages.append(HumanMessage(content=user_msg))
    #         if ai_msg:
    #             existing_messages.append(AIMessage(content=ai_msg))

    # # Rebuild message list: clear current messages, add old ones, then the latest human message
    # all_messages = [RemoveMessage(id=m.id) for m in messages] + existing_messages
    # if last_human_message:
    #     all_messages.append(HumanMessage(content=last_human_message.content))


    # return {
    #     "messages": all_messages,
    #     "thread_id": thread_id,
    # }
    return {
        "last_human_message_id": human_message_id
    }


async def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Process conversation through the language model with context."""
    
    messages = state.get("messages", [])
    summary = state.get("summary", "")

    prompt_parts = []

    # Build system context
    system_prompt = SystemMessage("""
        You are a helpful AI assistant designed to behave like a voice assistant (e.g., Alexa or Google Assistant).
        Always answer clearly, concisely, and in a natural conversational style.
        Prioritize providing direct, useful information without unnecessary elaboration.
        If you don't know the answer, say so plainly instead of inventing information.
        Follow user instructions carefully and avoid going off-topic.
        For multi-step or complex tasks, break responses into simple, actionable steps.
        Stay polite, neutral, and professional at all times.
        Do not include opinions, speculation, or personal experiences.
        Do not generate disallowed, unsafe, or harmful content.
        Avoid overly long answers unless explicitly requested.
        When clarification is needed, ask a short and direct follow-up question."""
    )
    prompt_parts.append(system_prompt)

    # if summary is available, add it to the system prompt
    if summary:
        rag_prompt = SystemMessage(f"""Here is the summary of the conversation so far: {summary}""")
        prompt_parts.append(rag_prompt)


    # extract last human message & the index of the last human message
    last_human_message = None
    last_human_message_index = None

    for i, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human_message = message
            last_human_message_index = i

    # print(f"Last human message index: {last_human_message_index}")
    # print(f"Last human message content: {last_human_message.content}")

    # Add last conversation history
    MAX_HISTORY_LENGTH = 2
    COUNT = 0
    recent_history = []

    # extract anything before last human_human_message_index till max_history_length considering the only the human messages will be covered when calculating the MAX_HISTORY_LENGTH
    for index, message in reversed(list(enumerate(messages[:last_human_message_index]))):
        
        if isinstance(message, HumanMessage):
            recent_history.append(message)
            COUNT += 1
            if COUNT == MAX_HISTORY_LENGTH:
                break
        else:
            recent_history.append(message)

    recent_history = reversed(recent_history)
    prompt_parts.extend(recent_history)



    # After adding the recent history, add the last human message and any flowwing messages like AI message containing tool calls or tool messages etc as well
    current_conversation = messages[last_human_message_index:]
    prompt_parts.extend(current_conversation)

    # format the prompt parts into a chat prompt
    chat_prompt = ChatPromptTemplate.from_messages(prompt_parts)
    # print(f"Chat prompt: {chat_prompt.format()}")
               
    # invoke the LLM with the chat prompt
    response = await llm_chat_model_with_tools.ainvoke(chat_prompt.messages, config=config)
    # print(f"LLM response: {response.content if response.content else response.tool_calls}")

    ai_message_id = await add_message(
        thread_id=config.get("configurable", {}).get("thread_id"), 
        role="assistant", 
        content=response.content
    )

    # if last message is AI message and tool calls are available, execute tools
    if isinstance(response, AIMessage) and getattr(response, "tool_calls", None):
        tool_calls = response.tool_calls

        for tool_call in tool_calls:
            await add_tool_call(
                message_id=ai_message_id,
                call_id=tool_call.get("id"),
                input_json=tool_call
            )

    return {
        "messages": [response], 
        "last_ai_message_id": ai_message_id
    }



async def generate_embeddings_for_query(message: str) -> List[float]:
    """Generate embeddings for the given query text."""
    return await embeddings_generator.aembed_query(message)


async def path_selector_post_llm_call(state: State, config: RunnableConfig) -> str:
    """Decide which path to take after LLM call."""
    messages = state.get("messages", [])

    # fail-safe, if no messages, end the workflow
    if not messages:
        return "workflow_completion"

    # get the last message
    last_message = messages[-1]

    # if last message is AI message and tool calls are available, execute tools
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools_execution"

    # if len(messages) is more than 2, summarize conversation
    if len(messages) > 2:
        return "summarize_conversation"

    # else end the workflow
    return "workflow_completion"


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

    return "call_model"


def retrieve_data_from_doc_RAG(state: State) -> Dict[str, Any]:
    """Placeholder for document RAG retrieval."""
    pass


def retrieve_data_from_web_RAG(state: State) -> Dict[str, Any]:
    """Placeholder for web RAG retrieval."""
    pass


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

    chat_messages = [
        {"role": "system", "content": summary_message}] + [
        {"role": "user" if getattr(m, "type", "") == "human" else "assistant", "content": getattr(m, "content", "")}
        for m in messages
    ]

    config = RunnableConfig(
        metadata={
            "thread_id": state.get("thread_id", ""),
            "source_application": "summarize_conversation"
        }
    )

    response = await llm_chat_model.ainvoke(chat_messages, config=config)

    return {"summary": getattr(response, "content", "")}


async def workflow_completion(state: State) -> Dict[str, Any]:
    """Workflow completion node."""
    return state