from langgraph.graph import StateGraph, START, END
from response_generation_agent.utils.state import AgentState  # Import AgentState
from response_generation_agent.agents import supervisor_agent, rag_agent, web_agent, email_writer_agent # Added email_writer_agent
from langchain_core.messages import HumanMessage

# Define a node to prepare the state with inquiry_summary converted to a message
def prepare_inquiry_node(state):
    """Prepares the state by adding inquiry_summary as a message."""
    # Extract the inquiry_summary from the state
    inquiry_summary = state.get("inquiry_summary", "")
    tone = state.get("tone", "")
    length = state.get("length", "")
    template = state.get("template", "")
    
    # Create a message containing the inquiry summary and other parameters
    inquiry_message = f"""Client Inquiry Summary:
{inquiry_summary}

Email Parameters:
- Tone: {tone}
- Length: {length}
- Template: {template}"""
    
    # Create a messages list with the inquiry
    if not state.get("messages"):
        # If no messages yet, add the inquiry as the first message
        state["messages"] = [HumanMessage(content=inquiry_message)]
    
    return state

# Define the graph
graph_builder = StateGraph(AgentState)  # Use AgentState here

# Add nodes for each agent, using string names as keys
graph_builder.add_node("prepare_inquiry", prepare_inquiry_node)  # Add preparation node
graph_builder.add_node("supervisor_agent", supervisor_agent, destinations=["rag_agent", "web_agent"])  # Original supervisor agent
graph_builder.add_node("rag_agent", rag_agent)
graph_builder.add_node("web_agent", web_agent)
graph_builder.add_node("email_writer_agent", email_writer_agent)

# Set the entry point to our preparation node
graph_builder.add_edge(START, "prepare_inquiry")

# After preparing the inquiry, go to the supervisor agent
graph_builder.add_edge("prepare_inquiry", "supervisor_agent")

# Edges from specialized agents back to the supervisor
graph_builder.add_edge("rag_agent", "supervisor_agent")
graph_builder.add_edge("web_agent", "supervisor_agent")

# Edge from supervisor_agent to email_writer_agent
graph_builder.add_edge("supervisor_agent", "email_writer_agent")

# The email_writer_agent is the final step
graph_builder.add_edge("email_writer_agent", END)

# Compile the graph
response_generation_graph = graph_builder.compile()