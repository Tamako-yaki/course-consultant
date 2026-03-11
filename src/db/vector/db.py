import os
from langchain_milvus import Milvus, BM25BuiltInFunction
from db.vector.embedding import EmbeddingModel
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = EmbeddingModel().get_embeddings()
    return _embeddings

class MilvusStore:
    
    def __init__(self):
        self.milvus_store = None
    
    async def ensure_connected(self, drop_old: bool = False):
        """在 async context 中初始化 Milvus 連線。
        不使用全域 vector_store 快取，以避免 Event Loop 關閉導致的錯誤。
        """
        if self.milvus_store is not None:
            return
            
        print("正在嘗試連接 Milvus 向量資料庫...")
        try:
            self.milvus_store = Milvus(
                embedding_function=_get_embeddings(),
                # builtin_function=BM25BuiltInFunction(),
                # vector_field=["dense", "sparse"],
                connection_args={
                    "host": os.getenv("MILVUS_HOST", "localhost"),
                    "port": os.getenv("MILVUS_PORT", "19530"),
                },
                collection_name=os.getenv("MILVUS_COLLECTION", "ntut_knowledge_base"),
                auto_id=True,
                drop_old=drop_old,
                enable_dynamic_field=True,
            )
            print("Milvus 向量資料庫連線成功。")
        except Exception as e:
            print(f"連線 Milvus 失敗: {e}")
            raise e
    
    async def get_retriever(self):
        await self.ensure_connected(drop_old=False)
        return self.milvus_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5, # 最後回傳 5 筆
                "fetch_k": 20, # 先從 Milvus 撈 20 筆候選
                "lambda_mult": 0.7, # 0~1 之間，越小越注重相似度，越大越注重多樣性
            },
        )
    
    async def get_multi_query_retriever(self, llm):
        await self.ensure_connected(drop_old=False)
        base_retriever = self.milvus_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 20,
                "fetch_k": 100,
                "lambda_mult": 0.7,
            },
        )
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
        )
    async def aadd_documents(self, documents, drop_old: bool = False):
        # 每次新增批次都重建 Milvus 實例，確保連線狀態與當前的 Event Loop 同步
        self.milvus_store = None 
        await self.ensure_connected(drop_old=drop_old)
        if self.milvus_store is not None:
            await self.milvus_store.aadd_documents(documents)   
            print(f"Batch of {len(documents)} documents indexed successfully.")
        else:
            print("Milvus 向量資料庫未連線，無法新增文件。")