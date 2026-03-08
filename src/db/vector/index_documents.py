from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from db.vector.store import MilvusStore

milvus_store = MilvusStore()

document_1 = Document(
    page_content="測試內容",
    metadata={"source": "dummy"},
)

document_2 = Document(
    page_content="測試內容",
    metadata={"source": "dummy"},
)

document_3 = Document(
    page_content="測試內容",
    metadata={"source": "dummy"},
)

docs = [document_1, document_2, document_3]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=256,
)

splitted_docs = text_splitter.split_documents(docs)

milvus_store.add_documents(splitted_docs)
print("Documents indexed successfully.")

