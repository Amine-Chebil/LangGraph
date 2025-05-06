from langgraph.prebuilt import create_react_agent
from response_generation_agent.utils.tools import rag_tool, TavilySearchResults, assign_to_rag_agent, assign_to_web_agent


rag_agent = create_react_agent(
	
    model="groq:meta-llama/llama-4-scout-17b-16e-instruct",
    tools=[rag_tool],
    prompt=(
        "You are a retrieval agent.\n\n"
        "INSTRUCTIONS:\n"
	"- Assist ONLY with hotel services related inquiries\n"
        "- Retrieve information that will help answering the guest inquiry\n"
        "- After you're done with your task, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="rag_agent",
)

web_agent = create_react_agent(
    model="groq:meta-llama/llama-4-scout-17b-16e-instruct",
    tools=[TavilySearchResults(max_results=1)],
    prompt=(
        "You are a web search agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with weather related inquiries\n"
	"- Retrieve information that will help answering the guest inquiry\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="web_agent",
)


supervisor_agent = create_react_agent(
    model="groq:meta-llama/llama-4-scout-17b-16e-instruct",
    tools=[assign_to_rag_agent, assign_to_web_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a rag agent. Assign hotel services related inquiries to this agent.\n"
        "- a web agent. Assign weather related inquiries to this agent\n"
	"In case you need one agent, assign work to Only the needed agent.\n"
        "In case you need both agents, Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    name="supervisor",
)