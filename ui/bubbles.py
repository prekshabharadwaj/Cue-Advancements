"""
ClueLy – ui/bubbles.py
Reusable chat bubble and indicator widgets.
"""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout
)

# ── Design tokens (shared) ────────────────────────────────────────────────────
COLOR_BG       = "#1a1b1e"
COLOR_SURFACE  = "#25262b"
COLOR_BORDER   = "#373a40"
COLOR_USER_BG  = "#5c7cfa"
COLOR_BOT_BG   = "#2c2e33"
COLOR_USER_TXT = "#ffffff"
COLOR_BOT_TXT  = "#c1c2c5"
COLOR_META     = "#5c5f66"
COLOR_ACCENT   = "#5c7cfa"
COLOR_INPUT_BG = "#2c2e33"
COLOR_SEND_BG  = "#5c7cfa"
COLOR_SEND_HV  = "#4c6ef5"
COLOR_TITLE    = "#e9ecef"
COLOR_REC      = "#fa5252"
COLOR_REC_DIM  = "#862e2e"


class MessageBubble(QFrame):
    """A single chat bubble – user (right, blue) or bot (left, dark)."""

    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.setObjectName("userBubble" if is_user else "botBubble")

        self._lbl = QLabel(text)
        self._lbl.setWordWrap(True)
        self._lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._lbl.setFont(QFont("Segoe UI", 10))
        self._lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.addWidget(self._lbl)

        if is_user:
            self.setStyleSheet(f"""
                QFrame {{
                    background: {COLOR_USER_BG};
                    border-radius: 16px;
                    border-bottom-right-radius: 4px;
                }}
                QLabel {{ color: {COLOR_USER_TXT}; background: transparent; }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    background: {COLOR_BOT_BG};
                    border-radius: 16px;
                    border-bottom-left-radius: 4px;
                    border: 1px solid {COLOR_BORDER};
                }}
                QLabel {{ color: {COLOR_BOT_TXT}; background: transparent; }}
            """)

        self.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )

    def append_text(self, text: str):
        """Stream tokens by appending to the label."""
        self._lbl.setText(self._lbl.text() + text)

    def set_text(self, text: str):
        self._lbl.setText(text)

    @property
    def text(self) -> str:
        return self._lbl.text()


class TypingIndicator(QFrame):
    """Animated '● ● ●' indicator shown while the LLM is generating."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = 0
        self._dots_lbl = QLabel("●  ●  ●")
        self._dots_lbl.setFont(QFont("Segoe UI", 9))
        self._dots_lbl.setObjectName("typingDots")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.addWidget(self._dots_lbl)

        self.setStyleSheet(f"""
            QFrame {{
                background: {COLOR_BOT_BG};
                border-radius: 16px;
                border-bottom-left-radius: 4px;
                border: 1px solid {COLOR_BORDER};
            }}
            QLabel#typingDots {{ color: {COLOR_ACCENT}; background: transparent; }}
        """)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(420)

    def _tick(self):
        self._frame = (self._frame + 1) % 3
        parts = ["●", "●", "●"]
        parts[self._frame] = "◉"
        self._dots_lbl.setText("  ".join(parts))

    def stop(self):
        self._timer.stop()


class SectionLabel(QFrame):
    """Thin separator label (e.g. '— Q3 —') between question–answer pairs."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFont(QFont("Segoe UI", 8))
        lbl.setStyleSheet(f"color: {COLOR_META}; background: transparent;")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.addWidget(lbl)
