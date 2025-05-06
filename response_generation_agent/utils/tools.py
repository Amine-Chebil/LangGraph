from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState
from langgraph.types import Command

# Load environment variables from .env file
load_dotenv()

# Set up your vector store and retriever
db = Chroma(persist_directory="chroma_db", embedding_function=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), collection_name="hotel")
retriever = db.as_retriever(search_kwargs={"k": 3})
rag_tool = create_retriever_tool(
    retriever,
    name="hotel_docs_search",
    description="Searches hotel-related documents for relevant information."
)

# Only TavilySearchResults and rag_tool are included

tools = [
    TavilySearchResults(max_results=3),
    rag_tool
]


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,  
            update={**state, "messages": state["messages"] + [tool_message]},  
            graph=Command.PARENT,  
        )

    return handoff_tool


# Handoffs
assign_to_rag_agent = create_handoff_tool(
    agent_name="rag_agent",
    description="Assign task to a rag agent.",
)

assign_to_web_agent = create_handoff_tool(
    agent_name="web_agent",
    description="Assign task to a web agent.",
)