import os
from dotenv import load_dotenv
load_dotenv()

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from db.vector.load_course_json import load_main_courses, load_standard_courses, load_syllabus_dir
from pathlib import Path
import asyncio
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings

MILVUS_HOST    = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT    = os.getenv("MILVUS_PORT", "19530")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "course_consultant")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "ntut_knowledge_base")
COLLECTION_DESCRIPTION = os.getenv("MILVUS_COLLECTION_DESCRIPTION", "無")

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"), 
        ("##", "h2"), 
        ("###", "h3")
    ],
    strip_headers=False, # 保留標題文字在 chunk 內，使語意更完整
)

txt_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=256,
)

def load_markdown_and_txt(dir_name: str) -> list[Document]:
    all_chunks: list[Document] = []
    root_dir = Path(__file__).parent.parent.parent.parent / "data" / dir_name
    print(f"讀取路徑: {root_dir}")
    if not root_dir.exists():
        print(f"Warning: 路徑 {root_dir} 不存在")
        return []

    for path in root_dir.rglob("*"):
        if not path.is_file() or path.suffix not in {".md", ".txt"}:
            continue
        print(f"處理文件: {path}")  
        text = path.read_text(encoding="utf-8")
        base_metadata = {
            "source":   str(path),
            "filename": path.name,
            "folder":   str(path.parent),
        }

        if path.suffix == ".md":
            chunks = md_splitter.split_text(text)
            for chunk in chunks:
                chunk.metadata = {**base_metadata, **chunk.metadata}
        else:  # .txt
            doc = Document(page_content=text, metadata=base_metadata)
            chunks = txt_splitter.split_documents([doc])

        all_chunks.extend(chunks)

    return all_chunks

def load_all_course_data() -> list[Document]:
    all_docs: list[Document] = []
    data_root = Path(__file__).parent.parent.parent.parent / "data" / "114的課程JSON（常潤提供）"
    
    # 1. 載入 main.json
    main_json_path = data_root / "1" / "main.json"
    all_docs.extend(load_main_courses(main_json_path))
    
    # 2. 載入 standard.json
    standard_json_path = data_root / "standard.json"
    standard_docs = load_standard_courses(standard_json_path)
    # standard 可能很大，需要切片
    standard_chunks = txt_splitter.split_documents(standard_docs)
    all_docs.extend(standard_chunks)
    
    # 3. 載入教學大綱
    syllabus_dir = data_root / "1" / "course"
    all_docs.extend(load_syllabus_dir(syllabus_dir))
    
    return all_docs

async def main():
    print("=== 開始讀取北科行政公開資料 ===")
    admin_chunks = load_markdown_and_txt("北科行政公開資料（常潤提供）")
    
    print("\n=== 開始讀取 114 課程 JSON 資料 ===")
    course_docs = load_all_course_data()
    
    all_chunks = admin_chunks + course_docs
    
    if not all_chunks:
        print("Warning: 沒有找到任何文件，請確認 data/ 資料夾內容")
    else:
        total_chunks = len(all_chunks)
        print(f"Total chunks to index: {total_chunks}")

        vector_store = Milvus(
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
            drop_old=True,
            enable_dynamic_field=True # 允許不同文件有不同的 metadata
        )
        print("Indexing documents to Milvus in batches...")
        
        batch_size = 500
        for i in range(0, total_chunks, batch_size):
            batch = all_chunks[i:i + batch_size]
            current_batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            print(f"[{current_batch_num}/{total_batches}] 正在處理第 {i} 到 {min(i + batch_size, total_chunks)} 筆資料...")
            
            # 第一批次時 drop_old=True，後續批次 drop_old=False
            is_first_batch = (i == 0)
            vector_store.add_documents(batch)

        print(f"Documents indexed successfully. ({total_chunks} chunks)")

if __name__ == "__main__":
    asyncio.run(main())