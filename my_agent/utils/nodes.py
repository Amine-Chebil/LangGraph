from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


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

# Define the summarization prompt
summarize_system_prompt = """You are an expert email summarizer for hotels specializing in extracting critical information.

TASK:
- Extract and summarize the key points from hotel guest emails

FOCUS ON:
- Reservation details (dates, room types, etc.)
- Specific requests, complaints, or inquiries
- Special requirements or expectations

FORMAT:
- concise key points
- Use clear, direct language
- Maintain factual accuracy
- Include all critical details for hotel staff to understand and categorize the request
- Avoid unnecessary details or redundant information

OUTPUT:
only the key points.

Your summary should allow hotel staff to quickly understand the email's purpose and required actions without reading the full message.
"""

# Define the summarization function
def summarize_email(state, config):
    """
    Summarizes the input email before classification.
    """
    
    email_body = state.get("email_body", "")
    if not email_body:
        print("Warning: No email body found to summarize.")
        return {} # No changes if no email body

    # Create messages for summarization
    messages = [
        SystemMessage(content=summarize_system_prompt),
        HumanMessage(content=email_body)
    ]

    # Get model configuration without binding tools for summarization
    model_name = config.get('configurable', {}).get("model_name", "groq")
    summarization_model = _get_model(model_name, bind_tools=False)
    
    # Invoke the summarization model
    response = summarization_model.invoke(messages)
    summary = response.content

    # Create a summary message
    summary_message = AIMessage(content=f"Email summary:\n{summary}")
    
    # Return the summary message to be added to state
    return {"messages": [summary_message]}

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


system_prompt = """You are an expert email classifier for hotel management systems with extensive understanding of hospitality operations.

TASK:
- First use the fetch_categories tool to get the complete list of available categories
- Analyze the provided email summary carefully
- Identify ALL relevant categories that apply to the email (can be multiple)
- Be precise and thorough in your classification

CLASSIFICATION GUIDELINES:
- Match content to the most specific categories available
- Consider both explicit requests and implied needs
- Classify by action required, not just keywords mentioned
- Multiple categories may apply to a single email

OUTPUT FORMAT:
only the classification result in JSON format:
- Format: ["category1", "category2", ...]
"""

# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    model_name = config.get('configurable', {}).get("model_name", "groq")
    
    # Add the system prompt for classification
    messages_with_system = [SystemMessage(content=system_prompt)] + messages
    
    model = _get_model(model_name)
    response = model.invoke(messages_with_system)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function to execute tools
tool_node = ToolNode(tools)