from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from response_generation_agent.utils.state import AgentState
from response_generation_agent.agents import supervisor_agent, rag_agent, web_agent
from langchain_core.messages import HumanMessage

# 1. prepare_inquiry — exactly as before
def prepare_inquiry_node(state):
    inquiry_summary = state.get("inquiry_summary", "")
    tone = state.get("tone", "")
    length = state.get("length", "")
    template = state.get("template", "")
    inquiry_message = f"""Client Inquiry Summary:
{inquiry_summary}

Email Parameters:
- Tone: {tone}
- Length: {length}
- Template: {template}
- Client name: Khlifa
- Hotel name: JAZ Hotel

Email Parameters Explained:
1. Length:
Under no circumstances should the length parameter cause hallucinations or unnecessary content.
Always keep the response accurate, concise, direct and fully aligned with the client’s request.
This parameter controls only verbosity, not content or creativity. 

2. Tone:
Formal: Formal language, no contractions, structured and respectful wording.
Professional: Professional vocabulary and structure; no casual expressions.
Friendly: Informal greetings, friendly phrasing and use contractions.
"""
    if not state.get("messages"):
        state["messages"] = [HumanMessage(content=inquiry_message)]
    return state

# 2. extractor node: invoke supervisor, grab structured_response, write to state
def supervisor_node(state) -> Command[str]:
    # invoke the React agent on the full conversation
    result = supervisor_agent.invoke({"messages": state["messages"]})
    # pull out the Pydantic object
    structured: any = result["structured_response"]
    # write it into state.email_response
    return Command(
        update={"email_response": structured},
        goto=END
    )

# 3. build the graph
graph_builder = StateGraph(AgentState)

# nodes
graph_builder.add_node("prepare_inquiry",    prepare_inquiry_node)
graph_builder.add_node("rag_agent",          rag_agent)
graph_builder.add_node("web_agent",          web_agent)
#graph_builder.add_node("supervisor_agent",   supervisor_agent)
graph_builder.add_node("supervisor", supervisor_node, destinations=["rag_agent", "web_agent"])

# linear edges (tools still loop back to supervisor)
graph_builder.add_edge(START,"prepare_inquiry")
graph_builder.add_edge("prepare_inquiry", "supervisor")
graph_builder.add_edge("rag_agent","supervisor")
graph_builder.add_edge("web_agent",       "supervisor")

# once supervisor_agent finishes, go extract_structured → END

graph_builder.add_edge("supervisor", END)

response_generation_graph = graph_builder.compile()