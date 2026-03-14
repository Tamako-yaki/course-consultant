from abc import ABC, abstractmethod
from enum import Enum
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict, Generator
from langgraph.graph import add_messages
import os

class BaseRAG(ABC):
    """RAG 基礎類別"""

    def __init__(self):
        self.llm = self._create_llm()
        self.memory = MemorySaver()
        self.graph = None

    def _create_llm(self):
        """建立並返回 LLM 實例"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("請在 .env 檔案中設置 GEMINI_API_KEY")
        
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
        )

    @abstractmethod
    def _build_graph(self):
        """建立圖結構"""
        pass
    
    def stream(self, user_message: str, session_id: str = "default") -> Generator[str, None, None]:
        """
        串流處理用戶消息
        
        Args:
            user_message: 使用者訊息
            session_id: 會話 ID。用於保持對話歷史

        Yields:
            串流的回應內容
        """
        pass