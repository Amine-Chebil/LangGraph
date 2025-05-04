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