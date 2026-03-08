from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from db.vector.store import MilvusStore
from agent.prompts import GENERATE_PROMPT
from agent.state import AgentState
from agent.configuration import Configuration

config = Configuration()

def retrieve(state: AgentState):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state
    Return:
        state (dict): The current graph state with documents
    """
    print("--- RETRIEVE ---")
    question = state["question"]
    retriever = MilvusStore().get_retriever()
    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} documents.")
    for i, doc in enumerate(docs):
        print(f"Document {i+1}: {doc.page_content[:200]}...")
    retrieved_docs = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_docs": retrieved_docs}

def generate(state: AgentState):
    """
    Generate answer based on retrieved documents

    Args:
        state (dict): The current graph state
    Return:
        state (dict): The current graph state with generation
    """
    print("--- GENERATE ---")
    question = state["question"]
    retrieved_docs = state["retrieved_docs"]
    conversation_history = state["messages"]

    formatted_prompt = GENERATE_PROMPT.format(
        question=question, 
        context=retrieved_docs,
        conversation_history=conversation_history
    )
    model = config.generate_model
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
    )
    response = llm.invoke(formatted_prompt)   
    return {
        "generation": response.content,
        "messages": [HumanMessage(content=question), AIMessage(content=response.content)],
    }
