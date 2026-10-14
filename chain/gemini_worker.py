"""
ClueLy – chain/gemini_worker.py
Background QThread worker that sends a transcribed question to
Google Gemini and streams tokens back to the UI.

Uses the new `google-genai` SDK (replaces deprecated `google-generativeai`).

Signals
-------
token_ready(str)    – each streamed text chunk (append to answer bubble)
answer_done()       – streaming complete
error_occurred(str) – on any failure
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from PySide6.QtCore import QThread, Signal

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are ClueLy, an expert real-time interview assistant.
The user is in a live interview. An interviewer has just asked the question below.
Provide a clear, concise, and impressive answer that the user can say out loud.
Structure your response for easy verbal delivery — use short paragraphs.
Do not use bullet points unless specifically helpful.
Be confident and professional.
"""

_CONTEXT_TEMPLATE = """\
{system}

--- Candidate Context ---
{context}
--- End Context ---

Interviewer's question: {question}
"""


class GeminiWorker(QThread):
    """
    Streams a Gemini response for a single interview question.
    Create a new instance for each question.
    """

    token_ready    = Signal(str)
    answer_done    = Signal()
    error_occurred = Signal(str)

    def __init__(
        self,
        question: str,
        context: str = "",
        model: str = "gemini-2.0-flash",
        parent=None,
    ):
        super().__init__(parent)
        self._question = question
        self._context  = context
        self._model    = model

    def run(self):
        try:
            from google import genai
            from google.genai import types

            api_key = os.getenv("GOOGLE_API_KEY", "").strip()
            if not api_key:
                self.error_occurred.emit(
                    "GOOGLE_API_KEY is not set in .env\n"
                    "Get one at: https://aistudio.google.com/app/apikey"
                )
                return

            client = genai.Client(api_key=api_key)

            # Build prompt
            if self._context.strip():
                prompt = _CONTEXT_TEMPLATE.format(
                    system=_SYSTEM_PROMPT,
                    context=self._context.strip(),
                    question=self._question,
                )
            else:
                prompt = (
                    f"{_SYSTEM_PROMPT}\n\n"
                    f"Interviewer's question: {self._question}"
                )

            # Stream response token-by-token
            for chunk in client.models.generate_content_stream(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=1024,
                ),
            ):
                if self.isInterruptionRequested():
                    break
                text = getattr(chunk, "text", None)
                if text:
                    self.token_ready.emit(text)

            self.answer_done.emit()

        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))
