from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModel:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            show_progress=False
        )

    def get_embeddings(self):
        return self.embeddings