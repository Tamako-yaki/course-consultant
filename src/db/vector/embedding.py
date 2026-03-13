from langchain_huggingface import HuggingFaceEmbeddings
import os

if os.getenv("HF_TOKEN") is None:
    raise ValueError("HF_TOKEN is not set.")
if os.getenv("EMBEDDING_DEVICE") is None:
    raise ValueError("EMBEDDING_DEVICE is not set. Please set it to your desired device (e.g., 'cpu', 'cuda', 'mps').")

class EmbeddingModel:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
            show_progress=False
        )

    def get_embeddings(self):
        return self.embeddings