import os
import logging
from dotenv import load_dotenv
load_dotenv()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import time
from typing import Annotated, List, Optional, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import add_messages, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from db.vector.db import MilvusStore
from .prompts import GENERATE_PROMPT
from .base_rag import BaseRAG

import transformers
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# 設置日誌級別，顯示生成的查詢
logging.basicConfig()
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)
    
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    retrieved_docs: List[str]
    final_answer: Optional[str]

class AdvancedRAG(BaseRAG):
    """
    Advanced RAG 進階檢索增強生成系統    
    """
    
    def __init__(self):
        """初始化 Advanced RAG"""
        super().__init__()
        self.vector_store = MilvusStore().store
        self.retriever = self._build_retriever()
        self.graph = self._build_graph()
        print("[AdvancedRAG] 初始化完成")
    
    def _build_retriever(self):
        """建立檢索器"""
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 50},
        )
        # 擴展查詢
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self.llm,
        )
        # 重排序
        compressor = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base"),
            top_n=20,
        )
        # 整合擴展查詢與重排序
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=multi_query_retriever,
        )
            
    def _retrieve_node(self, state: AgentState):
        """
        檢索相關文件，根據問題從 Milvus 撈取相關文件

        Args:
            state (dict): The current graph state
        Return:
            state (dict): The current graph state with documents
        """
        print("\n[RETRIEVE] 正在檢索相關文件...\n")

        original_query = ""
        
        if state["messages"]:
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    original_query = msg.content
                    break

        start_time = time.time()

        docs = self.retriever.invoke(original_query)
        
        end_time = time.time()
        
        print(f"檢索到 {len(docs)} 筆相關文件。")
        print(f"檢索花費 {end_time - start_time:.2f} 秒。")
        
        retrieved_docs = "\n\n".join([doc.page_content for doc in docs])

        return {
            "retrieved_docs": retrieved_docs,
        }

    async def _generate_node(self, state: AgentState):
        """
        生成回答，根據檢索到的文件來回答問題

        Args:
            state (dict): The current graph state
        Return:
            state (dict): The current graph state with generation
        """
        print("\n[GENERATE] 開始生成回答...\n")

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
        
        start_time = time.time()
        
        response = await self.llm.ainvoke(formatted_prompt)
        
        end_time = time.time()
        
        print(f"生成回應花費 {end_time - start_time:.2f} 秒.")

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
    
    async def stream(
        self,
        user_message: str,
        session_id: str = "default"
    ):
        """串流執行"""
        config = {
            "configurable": {"thread_id": session_id}
        }

        final_answer = None
        
        try:            
            async for event in self.graph.astream(
                {
                    "messages": [HumanMessage(content=user_message)],
                }, 
                config=config,
                stream_mode="values"
            ):
                if event.get("final_answer"):
                    final_answer = event["final_answer"]

            if final_answer:
                yield final_answer
            else:
                yield "抱歉，無法生成回覆。"

        except Exception as e:
            yield f"抱歉，發生錯誤: {str(e)}"