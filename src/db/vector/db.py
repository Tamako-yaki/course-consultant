from dotenv import load_dotenv
load_dotenv()
import os
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document

class MilvusStore:
    """
    Milvus 向量資料庫的封裝類別，提供連線、檢索和新增文件的功能。
    """

    def __init__(self):
        self._milvus_store: Milvus | None = None
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "ntut_knowledge_base")
        self.embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")
        self.embedding_model = "BAAI/bge-small-zh"
        self.reranking_model = "BAAI/bge-reranker-base"

    async def _init_vector_store(self, drop_old: bool = False) -> Milvus:
        print(f"正在連線至 Milvus (Host: {self.host}, Port: {self.port})...")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": self.embedding_device},
            show_progress=False
        )

        return Milvus(
            embedding_function=embeddings,
            # builtin_function=BM25BuiltInFunction(),
            # vector_field=["dense", "sparse"],
            connection_args={
                "host": self.host,
                "port": self.port,
            },
            collection_name=self.collection_name,
            auto_id=True,
            drop_old=drop_old,
            enable_dynamic_field=True, # 啟用動態欄位，允許不同文件有不同的 metadata 結構
        )

    async def get_store(self, drop_old: bool = False) -> Milvus:
        if self._milvus_store is None:
            try:
                self._milvus_store = await self._init_vector_store(drop_old=drop_old)
                print("Milvus 向量資料庫連線成功。")
            except Exception as e:
                print(f"連線 Milvus 失敗: {e}")
                raise e
        return self._milvus_store

    async def get_base_retriever(
        self, 
        k: int = 50
    ):
        store = await self.get_store()
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k, # 最後回傳 k 筆
                "fetch_k": k * 5, # 先從 Milvus 撈 k * 5 筆候選
                "lambda_mult": 0.7, # 0~1 之間，越小越注重相似度，越大越注重多樣性
            },
        )

    async def get_multi_query_retriever(self, llm, k: int = 50) -> MultiQueryRetriever:
        base_retriever = await self.get_base_retriever(k=k)
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
        )

    async def get_multi_query_rerank_retriever(self, llm, k: int = 20) -> ContextualCompressionRetriever:
        multi_query_retriever = await self.get_multi_query_retriever(llm, k*5)

        # 使用 CrossEncoder 進行重排序
        model = HuggingFaceCrossEncoder(model_name=self.reranking_model)
        compressor = CrossEncoderReranker(model=model, top_n=k)

        return ContextualCompressionRetriever(
            base_retriever=multi_query_retriever,
            base_compressor=compressor,
        )

    async def aadd_documents(self, documents: list[Document], drop_old: bool = False) -> None:
        store = await self.get_store(drop_old=drop_old)
        try:
            await store.aadd_documents(documents)
            print(f"成功新增 {len(documents)} 筆文件到 Milvus。")
        except Exception as e:
            print(f"新增文件到 Milvus 失敗: {e}")
            raise e

milvus_store = MilvusStore()