from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from email_analysis_agent.utils.nodes import classify_email, summarize_email, should_continue, tool_node
from email_analysis_agent.utils.state import AgentState


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai", "groq"]


# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the nodes we will cycle between
workflow.add_node("summarizer", summarize_email)
workflow.add_node("classifier", classify_email)
workflow.add_node("action", tool_node)

# Set the entrypoint as the summarizer
# This means that this node is the first one called
workflow.set_entry_point("summarizer")

# Add an edge from the summarizer to the agent
workflow.add_edge("summarizer", "classifier")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "classifier",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "classifier")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()
