from abc import ABC, abstractmethod
from enum import Enum
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, add_messages
from .base_rag import BaseRAG
from db.vector.db import MilvusStore
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
import time
from .prompts import GENERATE_PROMPT

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieved_docs: str
    
class NaiveRAG(BaseRAG):
    """簡易 RAG 系統"""

    def __init__(self):
        super().__init__()
        self.vector_store = MilvusStore().store
        self.graph = self._build_graph()

    def _retrieve_node(self, state: AgentState):
        """檢索相關文件"""
        print("[NaiveRAG] 檢索相關文件...")
        # 獲取最後一條用戶消息
        original_query = ""
        if state["messages"]:
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    original_query = msg.content
                    break
        print(f"[NaiveRAG] 檢索相關文件: {original_query}")
        start_time = time.time()
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 20, # 最後回傳 20 筆
            },
        )
        docs = retriever.invoke(original_query)
        end_time = time.time()
        print(f"[NaiveRAG] 檢索相關文件完成，耗時 {end_time - start_time:.2f} 秒。")
        retrieved_docs = "\n\n".join([doc.page_content for doc in docs])
        return {
            "retrieved_docs": retrieved_docs,
        }
    
    def _generate_node(self, state: AgentState):
        """生成回答"""
        print("[NaiveRAG] 生成回答...")
        # 獲取最後一條用戶訊息
        original_query = ""
        if state["messages"]:
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    original_query = msg.content
                    break
        retrieved_docs = state["retrieved_docs"]
        # 構建對話歷史
        conversation_history = state["messages"]
        formatted_prompt = GENERATE_PROMPT.format(
            original_query=original_query, 
            context=retrieved_docs,
            conversation_history=conversation_history
        )
        print(f"[NaiveRAG] 生成回答: {original_query}")
        start_time = time.time()
        response = self.llm.invoke(formatted_prompt)
        end_time = time.time()
        print(f"[NaiveRAG] 生成回答完成，耗時 {end_time - start_time:.2f} 秒。")
        return {
            "final_answer": response.content,
            "messages": [AIMessage(content=response.content)]
        }

    def _build_graph(self):
        """建立圖結構"""
        builder = StateGraph(AgentState)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("generate", self._generate_node)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile(checkpointer=self.memory)

    def stream(self, user_message: str, session_id: str = "default"):
        """流式處理用戶消息"""
        config = {
            "configurable": {"thread_id": session_id}
        }
        try:
            for event in self.graph.stream(
                {
                    "messages": [HumanMessage(content=user_message)],
                }, 
                config=config,
                stream_mode="values"
            ):
                if event.get("final_answer"):
                    yield event["final_answer"]
                else:
                    yield "抱歉，無法生成回覆。"
        except Exception as e:
            yield f"抱歉，發生錯誤: {str(e)}"