import os
from dotenv import load_dotenv
load_dotenv()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from pymilvus import Collection, MilvusException, connections, db, utility
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

MILVUS_HOST    = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT    = os.getenv("MILVUS_PORT", "19530")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "course_consultant")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "ntut_knowledge_base")
COLLECTION_DESCRIPTION = os.getenv("MILVUS_COLLECTION_DESCRIPTION", "無")

_vector_store = None

class MilvusStore:
    
    def __init__(self):
        self.store = self._get_vector_store()
    
    def _get_vector_store(self):
        global _vector_store

        if _vector_store is None:
            # 確保 database 存在
            try:
                connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
                existing_databases = db.list_database()

                if MILVUS_DB_NAME in existing_databases:
                    print(f"[DB] '{MILVUS_DB_NAME}' 已存在。")
                    db.using_database(MILVUS_DB_NAME)
                    for collection_name in utility.list_collections():
                        collection = Collection(name=collection_name)
                        print(f"[DB] 集合名稱: {collection_name}")
                        print(f"[DB] 集合描述: {collection.description}")
                else:
                    print(f"[DB] '{MILVUS_DB_NAME}' 不存在，正在建立...")
                    db.create_database(MILVUS_DB_NAME)
                    print(f"[DB] '{MILVUS_DB_NAME}' 建立成功。")

            except MilvusException as e:
                print(f"[DB] 初始化失敗: {e}")
                raise

            # 確保 collection 已載入記憶體
            if utility.has_collection(COLLECTION_NAME):
                Collection(COLLECTION_NAME).load()
                print(f"[DB] 集合 '{COLLECTION_NAME}' 已載入記憶體。")
            else:
                print(f"[DB] 集合 '{COLLECTION_NAME}' 不存在，請先建立集合。")

            # 建立 VectorStore
            print(f"[DB] 正在建立 VectorStore 連線到 '{MILVUS_DB_NAME}' 資料庫...")
            _vector_store = Milvus(
                embedding_function=HuggingFaceEmbeddings(
                    model_name="BAAI/bge-small-zh",
                    model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
                    show_progress=False,
                ),
                vector_field="dense",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 128},
                },
                search_params={
                    "metric_type": "COSINE",
                    "params": {"nprobe": 10},
                },
                connection_args={
                    "host": MILVUS_HOST,
                    "port": MILVUS_PORT,
                    "db_name": MILVUS_DB_NAME,
                },
                collection_name=COLLECTION_NAME,
                collection_description=COLLECTION_DESCRIPTION,
                consistency_level="Strong",
                auto_id=True,
                drop_old=False,
            )
            print(f"[DB] VectorStore 連線建立完成，集合名稱: '{COLLECTION_NAME}'")

        return _vector_store
        