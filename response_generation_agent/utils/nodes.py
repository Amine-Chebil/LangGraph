from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from response_generation_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import json
import re


@lru_cache(maxsize=4)
def _get_model(model_name: str, bind_tools: bool = True):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    elif model_name == "groq":
        model = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    if bind_tools:
        model = model.bind_tools(tools)
    return model
    
# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """Be a helpful assistant and generate a response to the user.
"""

# Define the function that calls the model
def generate_response(state, config):

    messages = state["messages"]
    model_name = config.get('configurable', {}).get("model_name", "groq")
    
    # Add the system prompt for classification
    messages_with_system = [SystemMessage(content=system_prompt)] + messages
    
    model = _get_model(model_name)
    response = model.invoke(messages_with_system)
    
    # Return message
    return {
        "messages": [response],
    }

# Define the function to execute tools
tool_node = ToolNode(tools)

supervisor_system_prompt = '''
You are a supervisor agent for a hotel guest support system. Your job is to:
1. Read the provided email summary describing the guest's inquiry.
2. Decide which information sources (tools) to use:
   - Use the hotel_docs_search tool for questions about hotel amenities, services, policies, or anything specific to the hotel.
   - Use the Tavily web search tool for questions about local events, weather, or general information not found in hotel documents.
   - Use both tools if the inquiry requires information from both sources.
3. After receiving the context/snippets from the selected tools, compose a single, helpful, and polite email response to the guest, using only the provided context.
4. If the provided context does not answer the guest's question, politely state that you are unable to answer based on the available information.

Always:
- Be concise and professional.
- Only use information from the provided context/snippets.
- Do not make up information.
'''

def supervisor_node(state, config):
    """
    Supervisor node that decides which agent(s) to call and composes the final response.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    messages = state["messages"]
    email_summary = state.get("email_summary", "")
    model_name = config.get('configurable', {}).get("model_name", "groq")

    # Check if there are tool responses in the messages
    tool_outputs = [m for m in messages if getattr(m, 'tool_call_id', None)]
    if tool_outputs:
        # If tool outputs exist, compose the final response using the context
        context = "\n".join([m.content for m in tool_outputs if hasattr(m, 'content')])
        final_prompt = f"You are a hotel guest support agent. Use ONLY the following context to answer the guest's inquiry.\nContext:\n{context}\nIf the context does not answer the question, politely say you are unable to answer.\n\nEmail summary: {email_summary}"
        supervisor_messages = [
            SystemMessage(content=final_prompt)
        ]
        model = _get_model(model_name, bind_tools=False)
        response = model.invoke(supervisor_messages)
        return {"messages": [response]}
    else:
        # First pass: decide which tools to call
        supervisor_messages = [
            SystemMessage(content=supervisor_system_prompt),
            HumanMessage(content=f"Email summary: {email_summary}")
        ]
        model = _get_model(model_name, bind_tools=True)
        response = model.invoke(supervisor_messages)
        return {"messages": [response]}