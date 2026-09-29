"""
ClueLy - Conversation Memory (LangChain v1)
Provides per-session in-memory chat history used by RunnableWithMessageHistory.
"""

from langchain_core.chat_history import InMemoryChatMessageHistory

# Session store: session_id -> InMemoryChatMessageHistory
_STORES: dict[str, InMemoryChatMessageHistory] = {}

DEFAULT_SESSION = "default"


def get_session_history(session_id: str = DEFAULT_SESSION) -> InMemoryChatMessageHistory:
    """Return (or create) the chat history store for a given session."""
    if session_id not in _STORES:
        _STORES[session_id] = InMemoryChatMessageHistory()
    return _STORES[session_id]


def clear_memory(session_id: str = DEFAULT_SESSION) -> None:
    """Wipe conversation history for a session."""
    if session_id in _STORES:
        _STORES[session_id].clear()
