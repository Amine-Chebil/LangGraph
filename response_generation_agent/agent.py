from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from response_generation_agent.utils.nodes import generate_response, should_continue, tool_node
from response_generation_agent.utils.state import AgentState


# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai", "groq"]

# Define a new graph
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Define the nodes we will cycle between
workflow.add_node("Generate Response", generate_response)

workflow.add_node("Tools", tool_node)

# Set the entrypoint
# This means that this node is the first one called
workflow.set_entry_point("Generate Response")

# We now add a conditional edge
workflow.add_conditional_edges(
    "Generate Response",
    should_continue,

    {
        # If `tools`, then we call the tool node.
        "continue": "Tools",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("Tools", "Generate Response")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
graph = workflow.compile()
