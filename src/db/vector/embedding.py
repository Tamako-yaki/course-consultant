from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModel:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cuda"},
            show_progress=False
        )

    def get_embeddings(self):
        return self.embeddings