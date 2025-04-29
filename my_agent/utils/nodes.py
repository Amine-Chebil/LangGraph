from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o")
    elif model_name == "anthropic":
        model = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    elif model_name == "groq":
        model = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

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


system_prompt = """You are an expert email classifier for hotels.

TASK:
- First use the fetch_categories tool to get available categories
- Analyze the email content carefully
- Classify the email into one or more relevant categories from the provided list
- Select ALL categories that apply to the email

GUIDELINES:
- Be precise in your classifications
- If an email spans multiple categories, include all relevant ones
- Focus on the content and intent of the email, not just keywords

OUTPUT FORMAT:
- Provide your final classification as a JSON array of strings
- Example: ["category1", "category2"]
"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get('configurable', {}).get("model_name", "groq")  # Changed default from "anthropic" to "groq"
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
tool_node = ToolNode(tools)