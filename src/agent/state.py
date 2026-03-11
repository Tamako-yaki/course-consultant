from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    question: str
    generation: str
    retrieved_docs: List[str]
    messages: Annotated[List[BaseMessage], add_messages]