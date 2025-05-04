from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool

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