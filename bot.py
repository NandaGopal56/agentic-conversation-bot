from typing import Dict, Any, List, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver as CheckpointMemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from storage import get_messages, save_message


load_dotenv()

class State(MessagesState):
    """State class that extends MessagesState to include summary"""
    summary: str
    thread_id: int
    doc_rag_results: str
    serach_rag_results: str


llm_chat_model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
embeddings_generator = OpenAIEmbeddings(model="text-embedding-ada-002")
memory_saver = CheckpointMemorySaver()

def memory_state_update(state: State) -> Dict[str, List[AIMessage]]:
    '''Update the memory state'''
    #print("Starting memory state update")
    #print(f"Initial state: {state}")
    #print(f"Initial message count: {len(state.get('messages', []))}")

    thread_history = get_messages(state.get("thread_id"))
    last_human_message = state.get("messages")[-1]
    #print(f"Processing last human message: {last_human_message}")
    
    thread_id = state.get("thread_id")
    #print(f"Processing for thread ID : {thread_id}")
        
    # Fetch messages from storage
    existing_messages = []
    summary = ""

    if thread_history:
        for msg_pair in thread_history:
            if msg_pair['user_message']:
                existing_messages.append(HumanMessage(content=msg_pair['user_message']))
            if msg_pair['ai_message']:
                existing_messages.append(AIMessage(content=msg_pair['ai_message']))
            if msg_pair['summary']:
                summary = msg_pair['summary']
                #print(f"Retrieved summary: {summary}")

    #print(f'Update state with messages')
    # Update state with messages, First delete any messages in state as existing_messages already contains everything that is fetched from storage
    all_messages = [RemoveMessage(id=m.id) for m in state["messages"]] \
        + existing_messages \
        + [HumanMessage(content=last_human_message.content)]
    
    new_state = {
        "summary": summary,
        "messages": all_messages
    }

    #print(f"Memory state update complete. New message count: {len(all_messages)}")
    # print(f"Final state after memory update: {new_state}")
    return new_state

def call_model(state: State) -> Dict[str, List[AIMessage]]:
    """Call the llm_chat_model with the current state"""
    #print("Starting llm_chat_model call")
    #print(f"Current message count: {len(state['messages'])}")
    #print(f"Current state: {state}")

    system_prompt = (
        "You are a helpful AI assistant. "
        "Answer all questions to the best of your ability."
        "if you are not aware of any information directly say so instead of making things up on your own"
    )

    # Start building the system message content
    system_message_content = system_prompt

    # Add summary if it exists
    if state.get("summary"):
        #print(f"Including summary in system message: {state['summary']}")
        system_message_content += (
            "\n\nThe provided chat history includes a summary of the earlier conversation.\n"
            f"Summary of conversation earlier: {state['summary']}"
        )

    # Add RAG results if available
    if state.get("doc_rag_results"):
        #print("Including RAG results in system message.")
        system_message_content += (
            "\n\nThe following context has been retrieved from relevant documents. "
            "Use this information to help answer the user's question accurately:\n"
            f"{state['doc_rag_results']}"
        )

    if state.get("web_rag_results"):
        #print("Including web RAG results in system message.")
        system_message_content += (
            "\n\nThe following context has been retrieved from web search. "
            "Use this information to help answer the user's question accurately:\n"
            f"{state['web_rag_results']}"
        )

    # Create the system message
    system_message = SystemMessage(content=system_message_content)

    # Combine messages
    question = [system_message] + state["messages"]
    #print(f"Prepared question for llm_chat_model: {question}") #TODO: revisit the promp. The human message is not being told as human message that the model needs to answer for correctly

    # Generate and stream response
    #print("Starting to stream response from llm_chat_model")
    full_response = ""
    
    #print('state before response', state)
    
    # Create an generator for the response
    print(f"Question: {question}")
    response = llm_chat_model.invoke(question)

    save_message(state.get("thread_id"), state["messages"][-1].content, response.content, state.get("summary", ""))

    return {
        "messages": [response]
    }
    #print("Finished streaming response from llm_chat_model")
    #print(f"Full response length: {len(full_response)} characters")

    #print('state after response', state)


def generate_embeddings_for_query(message: List[str]) -> List[List[float]]:
    vectors = embeddings_generator.embed_query(message)
    #print(f"Generated embeddings for query: {vectors}")
    return vectors


def should_continue(
    state: State
) -> str:

    # Get the number of messages in the conversation
    message_count: int = len(state["messages"])
    
    # Determine the next node to execute based on the message count
    decision: str = "summarize_conversation" if message_count > 2 else END

    # Log the decision-making process for debugging and monitoring
    #print(f"Workflow decision check - Messages: {message_count}")
    #print(f"Decision: {decision}")
    #print(f"Current state at decision point: {state}")

    # Return the determined next node
    return decision

def branch_selection_for_RAG(
    state: State
) -> Union[str, List[str]]:

    doc_enabled = state.get("is_doc_rag_enabled", False) #TODO: Fix this
    web_enabled = state.get("is_search_enabled", False) #TODO: Fix this

    if doc_enabled and web_enabled:
        return ["doc_rag_search", "web_rag_search"]
    elif doc_enabled:
        return "doc_rag_search"
    elif web_enabled:
        return "web_rag_search"
    else:
        return "conversation"
    
def retrieve_data_from_doc_RAG(
    state: State
) -> Dict[str, Any]:

    #print("[DEBUG] Running document RAG search...")
    last_human_message = [state['messages'][-1].content]

    vectors = [] #TODO: implement document RAG

    # Simulate document retrieval and update state
    results = [] #TODO: implement document RAG
    retrieved_docs = '\n'.join(item['data'] for item in results)
    return {"doc_rag_results": retrieved_docs}

def retrieve_data_from_web_RAG(
    state: State
) -> State:

    #print("[DEBUG] Running web RAG search...")
    # Simulate web search and update state
    retrieved_results = [] #TODO: implement web search
    state["web_rag_results"] = retrieved_results
    #print(f"[DEBUG] Retrieved from web RAG: {retrieved_results}")

    return {"web_rag_results": retrieved_results}


def summarize_conversation(state: State) -> Dict[str, Any]:
    """Summarize the conversation state"""
    #print("Starting conversation summarization")
    #print(f"Current message count: {len(state['messages'])}")
    #print(f"State before summarization: {state}")

    summary = state.get("summary", "")
    if summary:
        #print(f"Existing summary found: {summary}")
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        #print("No existing summary found, creating new summary")
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm_chat_model.invoke(messages)
    
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    
    final_state = {"summary": response.content, "messages": delete_messages}
    #print("Summarization complete")
    #print(f"Final message count: {len(delete_messages)}")
    #print(f"Final state after summarization: {final_state}")
    
    return final_state

def build() -> StateGraph:
    workflow = StateGraph(State)

    workflow.add_node("memory_state_update", memory_state_update)
    workflow.add_node("doc_rag_search", retrieve_data_from_doc_RAG)
    workflow.add_node("web_rag_search", retrieve_data_from_web_RAG)
    workflow.add_node("conversation", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)

    # Start
    workflow.add_edge(START, "memory_state_update")

    # Conditional branching to parallel paths
    workflow.add_conditional_edges("memory_state_update", branch_selection_for_RAG)

    # Both RAG paths must lead to conversation
    workflow.add_edge("doc_rag_search", "conversation")
    workflow.add_edge("web_rag_search", "conversation")

    # Conversation branching
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    workflow = workflow.compile(checkpointer=memory_saver)
    return workflow


def pretty_print_stream_chunk(chunk):
    for node, updates in chunk.items():
        print(f"Update from node: {node}")
        if "messages" in updates:
            updates["messages"][-1].pretty_print()
        else:
            print(updates)

        print("\n")


graph = build()

messages = [
    'Hello',
    'My name is Gopal',
    'What is my name'
]
for message in messages:
    for chunk in graph.stream({"messages": [message], "thread_id": 1}, config={"configurable": {"thread_id": 1}}):
        pretty_print_stream_chunk(chunk)