import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load environment variables from .env if available.
load_dotenv()

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
JSON_PATH = os.path.join(DATA_DIR, "114/standard.json")
ADMIN_DATA_DIR = os.path.join(DATA_DIR, "行政資料")
INDEX_SAVE_PATH = os.path.join(DATA_DIR, "faiss_index/pure_rag_ollama")
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


def load_standard_json(file_path: str) -> List[Document]:
    print(f"Loading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents: List[Document] = []
    for category, depts in data.items():
        for dept_name, content in depts.items():
            courses = content.get("courses", [])
            for course in courses:
                text = (
                    f"類別: {category} | 系所: {dept_name} | "
                    f"課程名稱: {course.get('name')} | 學分: {course.get('credit')} | "
                )
                metadata = {"source": "standard.json", "type": "course"}
                documents.append(Document(page_content=text, metadata=metadata))
    return documents


def load_admin_data(directory: str) -> List[Document]:
    print(f"Loading admin data from {directory}...")
    documents: List[Document] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt") or file.endswith(".md"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    metadata = {"source": os.path.basename(file_path), "type": "administrative"}
                    documents.append(Document(page_content=content, metadata=metadata))
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    return documents


def main():
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    if not ollama_base_url:
        raise ValueError("Missing OLLAMA_BASE_URL. Set it in your .env file.")

    print(f"Using Ollama embeddings model: {OLLAMA_EMBEDDING_MODEL}")
    print(f"Using Ollama base URL: {ollama_base_url}")

    admin_docs = load_admin_data(ADMIN_DATA_DIR)
    course_docs = load_standard_json(JSON_PATH)

    docs = admin_docs + course_docs
    print(f"Total documents: {len(docs)} (Admin: {len(admin_docs)}, Course: {len(course_docs)})")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    split_docs = text_splitter.split_documents(docs)
    print(f"Total chunks: {len(split_docs)}")

    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=ollama_base_url)

    vectorstore = None
    vectorstore_lock = threading.Lock()
    batch_size = 50
    batches = [split_docs[p : p + batch_size] for p in range(0, len(split_docs), batch_size)]

    def process_batch(idx_batch):
        idx, batch = idx_batch
        nonlocal vectorstore
        max_retries = 3

        for attempt in range(max_retries):
            try:
                tmp_vs = FAISS.from_documents(batch, embeddings)
                with vectorstore_lock:
                    if vectorstore is None:
                        vectorstore = tmp_vs
                    else:
                        vectorstore.merge_from(tmp_vs)
                if (idx + 1) % 10 == 0:
                    print(f"Batch {idx + 1}/{len(batches)} done.")
                return
            except Exception as e:
                print(f"Batch {idx + 1} attempt {attempt + 1} failed: {e}")
                time.sleep(2 * (attempt + 1))

    print(f"Processing {len(batches)} batches with 3 workers...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(process_batch, enumerate(batches))

    if vectorstore:
        print(f"Saving to {INDEX_SAVE_PATH}...")
        os.makedirs(os.path.dirname(INDEX_SAVE_PATH), exist_ok=True)
        vectorstore.save_local(INDEX_SAVE_PATH)
        print("Ingestion complete.")
    else:
        print("Ingestion failed: no vectors were created.")


if __name__ == "__main__":
    main()
