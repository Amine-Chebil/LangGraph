from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from response_generation_agent.utils.nodes import supervisor_node, should_continue, tool_node
from response_generation_agent.utils.state import AgentState

class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai", "groq"]

workflow = StateGraph(AgentState, config_schema=GraphConfig)

# Add supervisor as entry point
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Tools", tool_node)

workflow.set_entry_point("Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    should_continue,
    {
        "continue": "Tools",
        "end": END,
    },
)

workflow.add_edge("Tools", "Supervisor")

graph = workflow.compile()
