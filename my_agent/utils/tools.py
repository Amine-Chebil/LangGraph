from langchain_community.tools.tavily_search import TavilySearchResults

#tools = [TavilySearchResults(max_results=1)]

from langchain.tools import tool
from typing import List

@tool
def fetch_categories() -> List[str]:
    """Fetch relevant email categories for hotel management.
        
    Returns:
        A list of relevant categories for hotel email classification
    """
    # In production, this would fetch from a database or API
    # For now returning static categories
    return [
     "Spa",
    "Religion",
    "Group request",
    "Connecting rooms",
    "Disabled",
    "Bar",
    "Parking",
    "Specific room request",
    "Quiet room",
    "Jobs",
    "Boat dock",
    "Balcony",
    "Baby",
    "Reservation modification",
    "Reservation cancellation",
    "Complaint",
    "Reservation confirmation",
    "Flight + hotel",
    "Long Stay",
    "Coworking",
    "Day use",
    "Shuttle",
    "Store luggage",
    "Social request",
    "Rooftop",
    "Restaurant",
    "Taxi",
    "Celebration",
    "Mice",
    "Concierge",
    "Housekeeping",
    "Breakfast",
    "Smoker",
    "Travel agent",
    "Bedding pillow mattress",
    "Special request",
    "Check out",
    "Payment method",
    "Discount",
    "Rentals",
    "Special diet",
    "Family",
    "Reservation",
    "Contact staff",
    "Fallback",
    "Gym",
    "Tennis",
    "Other"
    ]

# Update the tools list
tools = [fetch_categories]