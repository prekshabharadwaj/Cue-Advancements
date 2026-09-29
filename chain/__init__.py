"""
ClueLy - LangChain integration package.

Public surface:
  from chain.qa_chain import build_qa_chain
  from chain.vectorstore import VectorStoreManager
  from chain.loader import load_documents
"""


def __getattr__(name):
    """Lazy-load chain submodules on first access to avoid import-time errors."""
    if name == "load_documents":
        from chain.loader import load_documents
        return load_documents
    if name == "VectorStoreManager":
        from chain.vectorstore import VectorStoreManager
        return VectorStoreManager
    if name == "build_qa_chain":
        from chain.qa_chain import build_qa_chain
        return build_qa_chain
    raise AttributeError(f"module 'chain' has no attribute {name!r}")


__all__ = ["load_documents", "VectorStoreManager", "build_qa_chain"]
