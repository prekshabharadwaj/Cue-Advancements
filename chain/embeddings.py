"""
ClueLy - Embeddings
Provides a single configured embeddings instance used by the vector store.
Model is controlled via the OPENAI_EMBEDDING_MODEL env var (default: text-embedding-3-small).
"""

import os
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Return a cached OpenAIEmbeddings instance."""
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
