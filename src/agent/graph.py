from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import expand_question, retrieve, generate

builder = StateGraph(AgentState)

builder.add_node("expand_question", expand_question)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "expand_question")
builder.add_edge("expand_question", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile(
    name="course-consultant",
    checkpointer=MemorySaver(),    
)
