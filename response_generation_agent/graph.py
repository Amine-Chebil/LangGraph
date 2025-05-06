from langgraph.graph import END
from langgraph.graph import StateGraph, START, MessagesState
from response_generation_agent.agents import supervisor_agent, rag_agent, web_agent

# Define the multi-agent supervisor graph
supervisor = (
    StateGraph(MessagesState)
    # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
    .add_node(supervisor_agent, destinations=("rag_agent", "web_agent", END))
    .add_node(rag_agent)
    .add_node(web_agent)
    .add_edge(START, "supervisor")
    # always return back to the supervisor
    .add_edge("rag_agent", "supervisor")
    .add_edge("web_agent", "supervisor")
    .compile()
)