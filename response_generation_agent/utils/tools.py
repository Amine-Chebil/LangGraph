from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from typing import Annotated
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.graph import MessagesState
from langgraph.types import Command, Send

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


def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help. You must provide a clear task_description."

    @tool(name, description=description)
    def handoff_tool(
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {"messages": [task_description_message]} 
        
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool


# Handoffs
assign_to_rag_agent = create_task_description_handoff_tool(
    agent_name="rag_agent",
    description="Assign a hotel services-related task to the RAG agent. Provide a clear task_description including all relevant details from the user's query.",
)

assign_to_web_agent = create_task_description_handoff_tool(
    agent_name="web_agent",
    description="Assign a web search task (e.g., for weather, general knowledge) to the web agent. Provide a clear task_description including the specific question or topic to search for.",
)

assign_to_email_writer_agent = create_task_description_handoff_tool(
    agent_name="email_writer_agent",
    description="Assign the task of writing a final email response to the email_writer_agent. Provide the original client inquiry summary and all gathered information as the task_description."
)