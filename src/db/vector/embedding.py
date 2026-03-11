from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

class EmbeddingModel:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={"device": "cuda"},
            show_progress=False
        )
        # self.embeddings = GoogleGenerativeAIEmbeddings(
        #     model="gemini-embedding-001",
        # )

    def get_embeddings(self):
        return self.embeddings