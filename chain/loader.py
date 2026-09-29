"""
ClueLy - Document Loader
Wraps LangChain's document loaders to ingest files from data/documents/.
Supported formats: .pdf, .txt, .md, .docx
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"

# Chunk settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

_LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".md":   UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
}


def load_documents(directory: Path = DOCUMENTS_DIR) -> List[Document]:
    """
    Load and chunk all supported documents from *directory*.

    Returns a flat list of LangChain Document chunks ready for embedding.
    """
    directory = Path(directory)
    raw: List[Document] = []

    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        loader_cls = _LOADER_MAP.get(path.suffix.lower())
        if loader_cls is None:
            continue
        try:
            loader = loader_cls(str(path))
            raw.extend(loader.load())
        except Exception as exc:  # noqa: BLE001
            print(f"[loader] Skipping {path.name}: {exc}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(raw)
