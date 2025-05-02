from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from email_analysis_agent.utils.tools import tools
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

# Define the summarization prompt
summarize_system_prompt = """You are an AI assistant specialized in hotel guest correspondence.  
Your task is to read each email and extract ONLY the main request(s) or question(s) about:
 - Hotel amenities and services  
 - Local area information  
 - Booking and reservation details  

Follow these rules:  
1. Begin each bullet with a strong verb (e.g. "Requests…", "Asks…", "Checks…").  
2. Preserve exact information from the email. 
3. **Use as few bullets as necessary**—omit anything not explicitly mentioned.  
4. Write formal, concise sentences.

Note: Your summary will be used to generate a response to the client, so ensure it is clear and accurate.
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
    
   # Return both the original email and the summary
    return {"messages": [HumanMessage(content=f"Original email:\n{email_body}"), summary_message],
            "email_summary": summary
            }
    

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


classify_system_prompt = """You are a multi-label classifier for hotel guest emails. 
your job is to identify the correct categories for the email.

TASK:
1. Get the list of available categories using the fetch_categories tool
2. Perform multi-label classification based on the available categories ONLY

Tools:
- fetch_categories: Get the list of available categories.

OUTPUT FORMAT:
Provide ONLY the result in JSON format:
- No labels → []
- One label → ["category"]
- Two or more labels → ["category1", "category2", ...]
"""

# Define the function that calls the model
def classify_email(state, config):

    messages = state["messages"]
    model_name = config.get('configurable', {}).get("model_name", "groq")
    
    # Add the system prompt for classification
    messages_with_system = [SystemMessage(content=classify_system_prompt)] + messages
    
    model = _get_model(model_name)
    response = model.invoke(messages_with_system)
    
    content = response.content
    categories = []
    try:
        # Look for JSON array in response
        match = re.search(r'\[(.*?)\]', content)
        if match:
            json_str = f"[{match.group(1)}]"
            categories = json.loads(json_str)
    except Exception:
        pass
    
    # Return both the message and update predicted_categories
    return {
        "messages": [response],
        "predicted_categories": categories  # Store categories in dedicated field
    }

# Define the function to execute tools
tool_node = ToolNode(tools)