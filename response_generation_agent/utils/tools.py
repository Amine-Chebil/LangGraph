from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.openweathermap.tool import OpenWeatherMapQueryRun
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

tools = [
    TavilySearchResults(max_results=1),
    OpenWeatherMapQueryRun()  # Will automatically use OPENWEATHERMAP_API_KEY from environment
]