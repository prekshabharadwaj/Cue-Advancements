# ClueLy

ClueLy is a floating, always-on-top notes widget with an integrated LangChain-powered
conversational QA engine. Drop documents into `data/documents/`, ask questions, and
keep your notes in the same window.

## Project Structure

```text
├── app.py                       # Entry point – launches the PySide6 window
│
├── ui/
│   ├── main_window.py           # Frameless floating window (opacity animation)
│   └── styles.py                # Theme constants
│
├── chain/                       # LangChain integration
│   ├── __init__.py              # Public API: load_documents, VectorStoreManager, build_qa_chain
│   ├── loader.py                # Document loaders  (PDF / TXT / MD / DOCX) + text splitter
│   ├── embeddings.py            # OpenAI embeddings (cached singleton)
│   ├── vectorstore.py           # FAISS index – build, persist, load, retrieve
│   ├── memory.py                # Windowed conversation buffer memory
│   └── qa_chain.py              # ConversationalRetrievalChain factory
│
├── storage/
│   └── notes.json               # Persisted note text + window geometry
│
├── data/
│   ├── documents/               # Drop source documents here (.pdf .txt .md .docx)
│   └── vectorstore/             # Auto-generated FAISS index (git-ignored)
│
├── .env                         # Local secrets (not committed)
├── .env.example                 # Template – copy to .env and fill in
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1 – Environment
```bash
cp .env.example .env
# Fill in OPENAI_API_KEY (and optionally OPENAI_CHAT_MODEL / OPENAI_EMBEDDING_MODEL)
```

### 2 – Install dependencies
```bash
pip install -r requirements.txt
```

### 3 – Add documents  *(optional)*
Drop `.pdf`, `.txt`, `.md`, or `.docx` files into `data/documents/`.

### 4 – Build the vector index  *(first run / after adding docs)*
```python
from dotenv import load_dotenv
load_dotenv()

from chain import load_documents, VectorStoreManager

vs = VectorStoreManager()
vs.build(load_documents())
print("Index built ✓")
```

### 5 – Run the app
```bash
python app.py
```

## Quick QA example
```python
from dotenv import load_dotenv
load_dotenv()

from chain import VectorStoreManager, build_qa_chain

vs = VectorStoreManager()
vs.load()

qa = build_qa_chain(vs)
result = qa.invoke({"question": "Summarise the main points of the uploaded documents."})
print(result["answer"])
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | – | **Required.** Your OpenAI API key. |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model used for answers. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for indexing. |
