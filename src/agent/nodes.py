import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from db.vector.db import MilvusStore
from agent.prompts import GENERATE_PROMPT 
from agent.state import AgentState
from agent.configuration import Configuration

milvus_store = MilvusStore()
config = Configuration()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set.")

async def retrieve(state: AgentState):
    """
    檢索相關文件，根據問題從 Milvus 撈取相關文件

    Args:
        state (dict): The current graph state
    Return:
        state (dict): The current graph state with documents
    """
    print("\n[RETRIEVE] 使用 MultiQueryRetriever 檢索...\n")
    question = state["question"]
    llm = ChatGoogleGenerativeAI(
        model=config.question_expand_model,
        temperature=0,
    )
    retriever = await milvus_store.get_multi_query_retriever(llm)
    docs = await retriever.ainvoke(question)
    print(f"檢索到 {len(docs)} 筆文件。")
    retrieved_docs = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_docs": retrieved_docs}

async def generate(state: AgentState):
    """
    生成回答，根據檢索到的文件來回答問題

    Args:
        state (dict): The current graph state
    Return:
        state (dict): The current graph state with generation
    """
    print("\n[GENERATE] 開始生成回答...\n")
    question = state["question"]
    retrieved_docs = state["retrieved_docs"]
    conversation_history = state["messages"]

    formatted_prompt = GENERATE_PROMPT.format(
        question=question, 
        context=retrieved_docs,
        conversation_history=conversation_history
    )
    llm = ChatGoogleGenerativeAI(
        model=config.generate_model,
        temperature=0,
    )
    start_time = time.time()
    response = await llm.ainvoke(formatted_prompt)
    end_time = time.time()
    print(f"生成回應花費 {end_time - start_time:.2f} 秒.")   
    return {
        "generation": response.content,
        "messages": [HumanMessage(content=question), AIMessage(content=response.content)],
    }
