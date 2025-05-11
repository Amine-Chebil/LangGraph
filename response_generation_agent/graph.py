from langgraph.graph import StateGraph, START, MessagesState
from response_generation_agent.agents import supervisor_agent, rag_agent, web_agent, email_writer_agent # Added email_writer_agent

# Define the graph
graph_builder = StateGraph(MessagesState)

# Add nodes for each agent, using their string names as keys
graph_builder.add_node("supervisor_agent", supervisor_agent)
graph_builder.add_node("rag_agent", rag_agent)
graph_builder.add_node("web_agent", web_agent)
graph_builder.add_node("email_writer_agent", email_writer_agent) # Added email_writer_agent node

# Set the entry point using the supervisor's string name
graph_builder.add_edge(START, "supervisor_agent")

# Edges from specialized agents back to the supervisor, using string names
graph_builder.add_edge("rag_agent", "supervisor_agent")
graph_builder.add_edge("web_agent", "supervisor_agent")

# The supervisor_agent's tools (assign_to_rag_agent, assign_to_web_agent, assign_to_email_writer_agent)
# will return Command(goto=[Send("rag_agent", ...)]), Command(goto=[Send("web_agent", ...)]),
# or Command(goto=[Send("email_writer_agent", ...)]).
# LangGraph uses these Send commands to route to the specified node names.
# No explicit conditional edges or destinations list is strictly needed on the supervisor_agent node itself
# for this Send-based routing to work, as long as the target node names are valid keys in the graph.

# The email_writer_agent is intended to be the final step for an inquiry.
# Its response will be the final output of the graph for that particular inquiry flow.
# Therefore, no outgoing edge is defined from "email_writer_agent" for this workflow.

# Compile the graph
response_generation_graph = graph_builder.compile()