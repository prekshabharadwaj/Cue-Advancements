"""
ClueLy - Chat Window
A floating, always-on-top chat widget that forwards questions to
LangChain (RAG via FAISS when docs exist, plain GPT otherwise).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from PySide6.QtCore import (
    Qt, QPoint, QTimer, QThread,
    QPropertyAnimation, QEasingCurve, QRect,
)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel,
    QFrame, QScrollArea, QSizePolicy,
)

from chain.worker import ChainWorker, PlainLLMWorker

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

OPACITY_IDLE  = 0.12
OPACITY_HOVER = 0.94
FADE_IN_MS    = 180
FADE_OUT_MS   = 400
HOVER_EXPAND  = 12
GROW_MS       = 180
SHRINK_MS     = 350

COLOR_BG       = "#1a1b1e"
COLOR_SURFACE  = "#25262b"
COLOR_BORDER   = "#373a40"
COLOR_USER_BG  = "#5c7cfa"
COLOR_BOT_BG   = "#2c2e33"
COLOR_USER_TXT = "#ffffff"
COLOR_BOT_TXT  = "#c1c2c5"
COLOR_META     = "#5c5f66"
COLOR_INPUT_BG = "#2c2e33"
COLOR_SEND_BG  = "#5c7cfa"
COLOR_SEND_HV  = "#4c6ef5"
COLOR_TITLE    = "#e9ecef"
COLOR_ACCENT   = "#5c7cfa"


class MessageBubble(QFrame):
    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.setObjectName("userBubble" if is_user else "botBubble")
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lbl.setFont(QFont("Segoe UI", 10))
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.addWidget(lbl)
        if is_user:
            self.setStyleSheet(f"""
                QFrame {{
                    background: {COLOR_USER_BG};
                    border-radius: 14px;
                    border-bottom-right-radius: 4px;
                }}
                QLabel {{ color: {COLOR_USER_TXT}; background: transparent; }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    background: {COLOR_BOT_BG};
                    border-radius: 14px;
                    border-bottom-left-radius: 4px;
                    border: 1px solid {COLOR_BORDER};
                }}
                QLabel {{ color: {COLOR_BOT_TXT}; background: transparent; }}
            """)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)


class TypingIndicator(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = 0
        self._dots_lbl = QLabel("●  ●  ●")
        self._dots_lbl.setFont(QFont("Segoe UI", 9))
        self._dots_lbl.setObjectName("typingDots")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.addWidget(self._dots_lbl)
        self.setStyleSheet(f"""
            QFrame {{
                background: {COLOR_BOT_BG};
                border-radius: 14px;
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


class ChatScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setStyleSheet(f"""
            QScrollArea {{ border: none; background: {COLOR_BG}; }}
            QScrollBar:vertical {{
                background: {COLOR_SURFACE}; width: 4px; border-radius: 2px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLOR_BORDER}; border-radius: 2px; min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)

    def scroll_to_bottom(self):
        QTimer.singleShot(30, lambda: self.verticalScrollBar().setValue(
            self.verticalScrollBar().maximum()
        ))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._drag_pos = QPoint()
        self._is_dragging = False
        self._hovered = False
        self._chain = None
        self._rag_ready = False
        self._worker_thread = None
        self._worker = None
        self._typing_widget = None
        self._typing_row = None

        self.setWindowTitle("ClueLy")
        self.setMinimumSize(340, 480)
        self.resize(400, 560)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(OPACITY_IDLE)
        self.setMouseTracking(True)
        self._build_ui()
        QTimer.singleShot(0, self._try_load_chain)
        self._opacity_anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._geo_anim = QPropertyAnimation(self, b"geometry", self)
        self._geo_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def _try_load_chain(self):
        try:
            from chain.vectorstore import VectorStoreManager
            from chain.qa_chain import build_qa_chain
            vs = VectorStoreManager()
            if vs.load():
                self._chain = build_qa_chain(vs)
                self._rag_ready = True
                self._set_mode_badge("RAG")
            else:
                self._set_mode_badge("GPT")
        except Exception as exc:
            self._set_mode_badge("ERR")
            print(f"[ClueLy] Chain init error: {exc}")

    def _set_mode_badge(self, mode: str):
        colours = {
            "RAG": ("#2f9e44", "#40c057"),
            "GPT": ("#1971c2", "#339af0"),
            "ERR": ("#c92a2a", "#fa5252"),
        }
        bg, fg = colours.get(mode, ("#444", "#aaa"))
        self._mode_badge.setText(f" {mode} ")
        self._mode_badge.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 5px;"
            f" font-size: 9px; font-weight: 700; padding: 1px 4px;"
        )

    def _build_ui(self):
        self._container = QFrame()
        self._container.setObjectName("container")
        self._container.setStyleSheet(f"""
            QFrame#container {{
                background: {COLOR_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 16px;
            }}
        """)

        title = QLabel("✦  ClueLy")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLOR_TITLE}; background: transparent;")

        self._mode_badge = QLabel(" … ")
        self._mode_badge.setFont(QFont("Segoe UI", 9))
        self._mode_badge.setStyleSheet(f"color: {COLOR_META}; background: transparent;")

        clear_btn = QPushButton("↺")
        clear_btn.setObjectName("clearBtn")
        clear_btn.setFixedSize(28, 26)
        clear_btn.setToolTip("Clear conversation")
        clear_btn.clicked.connect(self._clear_chat)
        clear_btn.setStyleSheet(f"""
            QPushButton#clearBtn {{
                background: transparent; color: {COLOR_META};
                border: none; border-radius: 5px; font-size: 16px;
            }}
            QPushButton#clearBtn:hover {{ background: #3c3f44; color: #ffffff; }}
        """)

        close_btn = QPushButton("×")
        close_btn.setObjectName("closeBtn")
        close_btn.setFixedSize(28, 26)
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet(f"""
            QPushButton#closeBtn {{
                background: transparent; color: {COLOR_META};
                border: none; border-radius: 5px; font-size: 18px;
            }}
            QPushButton#closeBtn:hover {{ background: #3c3f44; color: #ffffff; }}
        """)

        header = QHBoxLayout()
        header.setContentsMargins(14, 10, 10, 8)
        header.addWidget(title)
        header.addWidget(self._mode_badge)
        header.addStretch()
        header.addWidget(clear_btn)
        header.addWidget(close_btn)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"color: {COLOR_BORDER};")

        self._msg_container = QWidget()
        self._msg_container.setStyleSheet(f"background: {COLOR_BG};")
        self._msg_layout = QVBoxLayout(self._msg_container)
        self._msg_layout.setContentsMargins(10, 10, 10, 10)
        self._msg_layout.setSpacing(8)
        self._msg_layout.addStretch()

        self._scroll = ChatScrollArea()
        self._scroll.setWidget(self._msg_container)

        self._input = QTextEdit()
        self._input.setPlaceholderText("Ask anything…  (Enter to send, Shift+Enter for newline)")
        self._input.setFont(QFont("Segoe UI", 10))
        self._input.setFixedHeight(56)
        self._input.setAcceptRichText(False)
        self._input.setStyleSheet(f"""
            QTextEdit {{
                background: {COLOR_INPUT_BG}; color: {COLOR_TITLE};
                border: 1px solid {COLOR_BORDER}; border-radius: 10px; padding: 8px 10px;
            }}
            QTextEdit:focus {{ border: 1px solid {COLOR_ACCENT}; }}
        """)
        self._input.installEventFilter(self)

        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("sendBtn")
        self._send_btn.setFixedSize(60, 40)
        self._send_btn.clicked.connect(self._send_message)
        self._send_btn.setStyleSheet(f"""
            QPushButton#sendBtn {{
                background: {COLOR_SEND_BG}; color: #ffffff;
                border: none; border-radius: 10px; font-size: 12px; font-weight: 600;
            }}
            QPushButton#sendBtn:hover {{ background: {COLOR_SEND_HV}; }}
            QPushButton#sendBtn:disabled {{ background: #3c3f44; color: {COLOR_META}; }}
        """)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(10, 6, 10, 10)
        input_row.setSpacing(8)
        input_row.addWidget(self._input)
        input_row.addWidget(self._send_btn)

        body = QVBoxLayout(self._container)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addLayout(header)
        body.addWidget(div)
        body.addWidget(self._scroll, 1)
        body.addLayout(input_row)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.addWidget(self._container)

        self._add_bot_bubble(
            "Hi! I'm ClueLy. Ask me anything — I'll search your documents "
            "if they're indexed, or answer directly with GPT."
        )

    def _send_message(self):
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        self._add_user_bubble(text)
        self._show_typing()
        self._send_btn.setEnabled(False)

        self._worker_thread = QThread(self)
        if self._rag_ready and self._chain is not None:
            self._worker = ChainWorker(self._chain, text)
        else:
            self._worker = PlainLLMWorker(text)

        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.answer_ready.connect(self._on_answer)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.answer_ready.connect(self._worker_thread.quit)
        self._worker.error_occurred.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _on_answer(self, answer: str):
        self._hide_typing()
        self._add_bot_bubble(answer)
        self._send_btn.setEnabled(True)

    def _on_error(self, error: str):
        self._hide_typing()
        self._add_bot_bubble(f"⚠  Error: {error}")
        self._send_btn.setEnabled(True)

    def _add_user_bubble(self, text: str):
        bubble = MessageBubble(text, is_user=True)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(bubble)
        pos = self._msg_layout.count() - 1
        self._msg_layout.insertLayout(pos, row)
        self._scroll.scroll_to_bottom()

    def _add_bot_bubble(self, text: str):
        bubble = MessageBubble(text, is_user=False)
        row = QHBoxLayout()
        row.addWidget(bubble)
        row.addStretch()
        pos = self._msg_layout.count() - 1
        self._msg_layout.insertLayout(pos, row)
        self._scroll.scroll_to_bottom()

    def _show_typing(self):
        self._typing_widget = TypingIndicator()
        row = QHBoxLayout()
        row.addWidget(self._typing_widget)
        row.addStretch()
        self._typing_row = row
        pos = self._msg_layout.count() - 1
        self._msg_layout.insertLayout(pos, row)
        self._scroll.scroll_to_bottom()

    def _hide_typing(self):
        if self._typing_widget is not None:
            self._typing_widget.stop()
        if self._typing_row is not None:
            idx = self._msg_layout.indexOf(self._typing_row)
            if idx >= 0:
                item = self._msg_layout.takeAt(idx)
                if item and item.layout():
                    while item.layout().count():
                        child = item.layout().takeAt(0)
                        if child.widget():
                            child.widget().deleteLater()
        self._typing_widget = None
        self._typing_row = None

    def _clear_chat(self):
        while self._msg_layout.count() > 1:
            item = self._msg_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
        try:
            from chain.memory import clear_memory
            clear_memory()
        except Exception:
            pass
        self._add_bot_bubble("Conversation cleared. What would you like to know?")

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent
        if obj is self._input and event.type() == QEvent.Type.KeyPress:
            ke = QKeyEvent(event)
            if ke.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if ke.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    return False
                self._send_message()
                return True
        return super().eventFilter(obj, event)

    def enterEvent(self, event):
        self._hovered = True
        self._anim_opacity(OPACITY_HOVER, FADE_IN_MS)
        self._anim_geo(expand=True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._anim_opacity(OPACITY_IDLE, FADE_OUT_MS)
        self._anim_geo(expand=False)
        super().leaveEvent(event)

    def _anim_opacity(self, target: float, ms: int):
        self._opacity_anim.stop()
        self._opacity_anim.setDuration(ms)
        self._opacity_anim.setStartValue(self.windowOpacity())
        self._opacity_anim.setEndValue(target)
        self._opacity_anim.start()

    def _anim_geo(self, expand: bool):
        cur = self.geometry()
        d = HOVER_EXPAND if expand else -HOVER_EXPAND
        target = QRect(
            cur.x() - d, cur.y() - d,
            cur.width() + d * 2, cur.height() + d * 2,
        )
        self._geo_anim.stop()
        self._geo_anim.setDuration(GROW_MS if expand else SHRINK_MS)
        self._geo_anim.setStartValue(cur)
        self._geo_anim.setEndValue(target)
        self._geo_anim.start()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= 46:
            self._is_dragging = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_dragging and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._is_dragging = False
        super().mouseReleaseEvent(event)