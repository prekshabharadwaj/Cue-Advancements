"""
ClueLy - Background Worker
Runs the LangChain QA chain (or plain LLM) in a QThread so the UI
never freezes during inference.
"""

import os

from PySide6.QtCore import QObject, Signal


class ChainWorker(QObject):
    """
    Emits `answer_ready(str)` when inference completes,
    or `error_occurred(str)` on failure.
    """

    answer_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, chain, question: str, parent=None):
        super().__init__(parent)
        self._chain = chain
        self._question = question

    def run(self):
        try:
            result = self._chain.invoke(
                {"question": self._question},
                config={"configurable": {"session_id": "default"}},
            )
            # LCEL chain returns str directly
            self.answer_ready.emit(str(result))
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))


class PlainLLMWorker(QObject):
    """
    Falls back to a plain ChatOpenAI call (no RAG) when the vector
    store has not been initialised.
    """

    answer_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, question: str, parent=None):
        super().__init__(parent)
        self._question = question

    def run(self):
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            llm = ChatOpenAI(
                model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                temperature=0.2,
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            response = llm.invoke([HumanMessage(content=self._question)])
            self.answer_ready.emit(response.content)
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))
