"""
Pipeline utilities for enhanced RAG retrieval.

Includes query expansion and document reranking.
"""

import re
from typing import List
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


class QueryExpander:
    """
    Expand a query into hypothetical answer passages (HyDE) personalized with
    student context extracted from the conversation history.
    """

    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash-lite"):
        """
        Initialize QueryExpander with Gemini API.

        Args:
            api_key: Google API key for Gemini
            model: Model name to use (default: gemini-2.5-flash-lite)
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.7,
            max_output_tokens=600,
            google_api_key=api_key,
        )

        # Prompt: extract student context, then produce 3 HyDE document-chunk passages
        self.prompt = PromptTemplate(
            input_variables=["query", "chat_history"],
            template="""You are a retrieval assistant for a university course consultation system. The vector store contains policy documents, course catalogs, and academic regulations.

Internally (do not output this): note any student personal details from the conversation history (major, year, interests, GPA, etc.) and use them to make the passages more targeted.

Your ONLY output must be exactly 3 short passages that look like raw text chunks that would exist in the vector store and directly contain the answer to the student's query. Each passage should be 1–3 sentences, dense and factual, like a copied excerpt from a university regulation or course catalog — NOT a fluent answer, NOT a rephrased question, NO step labels, NO explanations.

Good examples:
- 英語畢業門檻：大學部學生須於畢業前達到 CEFR B2 等級，或通過校定英語能力鑑定考試。
- Scholarship eligibility: Full-time students maintaining a GPA above 3.5 with no failing grades in the preceding semester.
- 選課規定：大三以上學生每學期最多可修 25 學分，需經指導教授簽核。

Conversation History:
{chat_history}

Student Query: {query}

Output (3 passages, one per line, nothing else):""",
        )

    def expand_query(self, query: str, chat_history: str = "") -> List[str]:
        """
        Expand a query into 4 variants: original + 3 HyDE passages.

        Args:
            query: Original query string
            chat_history: Formatted conversation history string

        Returns:
            List of 4 items: [original_query, passage_1, passage_2, passage_3]
        """
        try:
            chain = self.prompt | self.llm
            response = chain.invoke({"query": query, "chat_history": chat_history or "無"})
            response_text = response.content.strip()

            # Each non-empty line is one passage
            passages = [line.strip() for line in response_text.split("\n") if line.strip()]

            result = [query] + passages[:3]
            return result if len(result) == 4 else [query, query, query, query]

        except Exception as e:
            print(f"Query expansion failed: {e}. Using original query.")
            return [query, query, query, query]


class DocumentReranker:
    """Rerank documents using BGE reranker model from HuggingFace."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Initialize DocumentReranker.

        Args:
            model_name: HuggingFace model name for reranker (default: BAAI/bge-reranker-base)
        """
        try:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install it with: pip install sentence-transformers"
            )
        except Exception as e:
            print(f"Failed to load reranker model {model_name}: {e}")
            raise

    def rerank(
        self, query: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        """
        Rerank documents by relevance to the query.

        Args:
            query: Original query string
            documents: List of Document objects to rerank
            top_k: Number of top documents to return (default: 5)

        Returns:
            List of top-k documents sorted by relevance score (highest first)
        """
        if not documents:
            return []

        # Extract document texts
        doc_texts = [doc.page_content for doc in documents]

        # Create pairs of (query, document) for scoring
        pairs = [[query, text] for text in doc_texts]

        # Get relevance scores
        scores = self.reranker.predict(pairs)

        # Create list of (document, score) tuples
        doc_score_pairs = list(zip(documents, scores))

        # Sort by score descending
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k documents (without scores)
        return [doc for doc, score in doc_score_pairs[:top_k]]
