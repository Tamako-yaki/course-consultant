import os
from langchain_milvus import Milvus
from db.vector.embedding import EmbeddingModel

_vector_store = None
embeddings = EmbeddingModel().get_embeddings()

class MilvusStore:
    
    def __init__(self):
        self.milvus_store = self.get_vector_store()
    
    def get_vector_store(self):
        global _vector_store
        if _vector_store is None:
            print("Milvus 向量資料庫尚未初始化，正在嘗試連接...")
            try:
                _vector_store = Milvus(
                    embedding_function=embeddings,
                    connection_args={
                        "host": os.getenv("MILVUS_HOST", "localhost"),
                        "port": os.getenv("MILVUS_PORT", "19530"),
                    },
                    auto_id=True,
                    drop_old=False,
                )
                print("Milvus 向量資料庫連線成功。")
            except Exception as e:
                print(f"連線 Milvus 失敗: {e}")
                return None
        return _vector_store
    
    def get_retriever(self):
        return self.milvus_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 20,
                #TODO: make source dynamic
                # "filter": {
                #     "metadata": {
                #         "$and": [
                #             {"source": {"$eq": "course_material"}},
                #         ]
                #     }
                # }
            },
        )
    
    def add_documents(self, documents):
        if self.milvus_store is not None:
            self.milvus_store.add_documents(documents)
        else:
            print("Milvus 向量資料庫未連線，無法新增文件。")