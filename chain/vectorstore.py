"""
ClueLy - FAISS Vector Store Manager
Handles building, persisting, loading, and querying the FAISS index.

Index is saved to  data/vectorstore/  so it survives restarts.
"""

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from chain.embeddings import get_embeddings


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "vectorstore"


class VectorStoreManager:
    """Thin wrapper around a FAISS index with disk persistence."""

    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        self.index_dir = Path(index_dir)
        self._store: Optional[FAISS] = None

    # ------------------------------------------------------------------
    # Build / load
    # ------------------------------------------------------------------

    def build(self, documents: List[Document]) -> None:
        """
        Create a new FAISS index from *documents* and persist it to disk.
        Replaces any existing index.
        """
        if not documents:
            raise ValueError("No documents provided — index not built.")

        self._store = FAISS.from_documents(documents, get_embeddings())
        self._persist()

    def load(self) -> bool:
        """
        Load an existing index from disk.
        Returns True on success, False if no saved index exists.
        """
        index_file = self.index_dir / "index.faiss"
        if not index_file.exists():
            return False

        self._store = FAISS.load_local(
            str(self.index_dir),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        return True

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to an already-loaded or already-built index."""
        if self._store is None:
            self.build(documents)
            return
        self._store.add_documents(documents)
        self._persist()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def as_retriever(self, k: int = 4):
        """Return a LangChain retriever for use inside a chain."""
        if self._store is None:
            raise RuntimeError("Vector store is not initialised. Call build() or load() first.")
        return self._store.as_retriever(search_kwargs={"k": k})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.index_dir))

    @property
    def is_ready(self) -> bool:
        return self._store is not None
