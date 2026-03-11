import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from db.vector.db import MilvusStore
from agent.prompts import QUESTION_EXPAND_PROMPT, GENERATE_PROMPT 
from agent.state import AgentState
from agent.configuration import Configuration
from agent.schemas import SubQuestions

milvus_store = MilvusStore()
config = Configuration()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set.")

async def expand_question(state: AgentState):
    """
    擴展問題，將複雜問題擴展成多個子問題
    
    Args:
        state (dict): The current graph state
    Return:
        state (dict): The current graph state with sub-questions
    """
    print("\n[EXPAND QUESTION] 開始擴展問題...\n")
    question = state["question"]
    conversation_history = state["messages"]

    formatted_prompt = QUESTION_EXPAND_PROMPT.format(
        question=question,
        conversation_history=conversation_history
    )

    model = config.decompose_model
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
    ).with_structured_output(SubQuestions)
    response = await llm.ainvoke(formatted_prompt)   
    sub_questions = response.sub_questions
    print(f"分解為 {len(sub_questions)} 個子問題。")
    for i, sub_q in enumerate(sub_questions):
        print(f"子問題 {i+1}: {sub_q}")
    return {"sub_questions": sub_questions}

async def retrieve(state: AgentState):
    """
    檢索相關文件，根據子問題從 Milvus 撈取相關文件，並去重

    Args:
        state (dict): The current graph state
    Return:
        state (dict): The current graph state with documents
    """
    print("\n[RETRIEVE] 開始從 Milvus 撈取相關文件...\n")
    sub_questions = state["sub_questions"]
    retriever = await milvus_store.get_retriever()

    all_results = []
    seen_ids = set()
    total_fetched = 0

    for q in sub_questions:
        docs = await retriever.ainvoke(q)
        total_fetched += len(docs)
        for doc in docs:
            doc_id = doc.metadata.get("pk", hash(doc.page_content)) # 嘗試用 Milvus 的 pk 作為唯一 ID，如果沒有就用內容的 hash
            if doc_id not in seen_ids: # 去重
                seen_ids.add(doc_id)
                all_results.append(doc) 
        print(f"問題: {q} -> 撈取 {len(docs)} 筆，累計獨立文件數: {len(all_results)}")
    print(f"總共從 Milvus 撈取 {total_fetched} 筆候選文件，去重後 {len(all_results)} 筆。")
    for i, doc in enumerate(all_results[:5]): # 只印前 5 筆
        print(f"文件 {i+1}: {doc.page_content[:100]}... (metadata: {doc.metadata})")
    retrieved_docs = "\n\n".join([doc.page_content for doc in all_results])
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
    model = config.generate_model
    llm = ChatGoogleGenerativeAI(
        model=model,
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
