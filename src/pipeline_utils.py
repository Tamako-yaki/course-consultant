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
    """Expand a single query into multiple variants using Gemini."""

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
            max_output_tokens=200,
            google_api_key=api_key,
        )

        # Prompt template for query expansion
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="""You are a query expansion expert. Given a user query, generate 2 alternative phrasings of the same query.

Original Query: {query}

Generate 2 short and concise alternative phrasings that capture the same intent but use different words and phrasing. Return only the 2 alternatives, one per line, without numbering or explanations.

Alternative 1:
Alternative 2:""",
        )

    def expand_query(self, query: str) -> List[str]:
        """
        Expand a single query into 3 variants.

        Args:
            query: Original query string

        Returns:
            List of 3 queries: [original_query, variant_1, variant_2]
        """
        try:
            # Get LLM response
            chain = self.prompt | self.llm
            response = chain.invoke({"query": query})
            response_text = response.content.strip()

            # Parse the two alternatives from the response
            lines = [line.strip() for line in response_text.split("\n") if line.strip()]

            # Extract non-empty lines
            alternatives = [line for line in lines if line and not line.startswith("Alternative")]

            # Return original query + 2 variants (or fewer if parsing fails)
            result = [query]
            result.extend(alternatives[:2])

            return result if len(result) == 3 else [query, query, query]

        except Exception as e:
            # Fallback: return original query 3 times if expansion fails
            print(f"Query expansion failed: {e}. Using original query.")
            return [query, query, query]


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
