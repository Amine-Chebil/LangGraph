from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    email_body: str
    email_summary: str  # Summarized email body
    available_categories: list[str]  # Available categories
    predicted_categories: list[str]  # Classification
