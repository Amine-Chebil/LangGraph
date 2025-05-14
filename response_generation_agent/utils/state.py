from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    inquiry_summary: str
    agents_responses: str
    draft_email: str
    tone: str
    length: str
    template: str
    email_response: str
