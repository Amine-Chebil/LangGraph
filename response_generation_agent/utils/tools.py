from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.openweathermap.tool import OpenWeatherMapQueryRun
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

tools = [
    TavilySearchResults(max_results=1),
    OpenWeatherMapQueryRun(),  # Will automatically use OPENWEATHERMAP_API_KEY from environment
    rag_tool
]