"""
ClueLy – ui/main_window.py
Floating, always-on-top interview assistant.

Flow:
  1. User clicks ▶ Record → system audio loopback starts
  2. Live transcript preview updates in real-time (interim STT)
  3. On silence pause OR manual ✉ Send → question bubble appears → Gemini streams answer
  4. User clicks ■ Stop → recording ends
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from PySide6.QtCore import (
    Qt, QPoint, QTimer, QPropertyAnimation, QEasingCurve, QRect,
)
from PySide6.QtGui import QFont, QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLabel, QFrame, QScrollArea, QSizePolicy, QGraphicsDropShadowEffect,
)

from ui.bubbles import (
    MessageBubble, TypingIndicator, SectionLabel,
    COLOR_BG, COLOR_SURFACE, COLOR_BORDER, COLOR_META,
    COLOR_TITLE, COLOR_ACCENT, COLOR_INPUT_BG,
    COLOR_SEND_BG, COLOR_SEND_HV, COLOR_REC, COLOR_REC_DIM,
    COLOR_BOT_BG, COLOR_BOT_TXT,
)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ── Animation constants ───────────────────────────────────────────────────────
OPACITY_IDLE  = 0.12
OPACITY_HOVER = 0.96
FADE_IN_MS    = 160
FADE_OUT_MS   = 380
HOVER_EXPAND  = 10
GROW_MS       = 160
SHRINK_MS     = 320


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


class RecordButton(QPushButton):
    """Pulsing red record button with animated glow when active."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recording = False
        self._pulse = 0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._do_pulse)
        self.setFixedSize(38, 38)
        self._apply_style(False)

    def set_recording(self, active: bool):
        self._recording = active
        if active:
            self._pulse_timer.start(600)
        else:
            self._pulse_timer.stop()
            self._pulse = 0
        self._apply_style(active)

    def _do_pulse(self):
        self._pulse = 1 - self._pulse
        self._apply_style(True, dim=self._pulse == 0)

    def _apply_style(self, active: bool, dim: bool = False):
        if active:
            bg = COLOR_REC_DIM if dim else COLOR_REC
            self.setText("■")
            self.setToolTip("Stop recording")
        else:
            bg = "#2c2e33"
            self.setText("▶")
            self.setToolTip("Start recording")
        self.setStyleSheet(f"""
            QPushButton {{
                background: {bg};
                color: {'#ff8080' if active else COLOR_META};
                border: 1px solid {'#fa5252' if active else COLOR_BORDER};
                border-radius: 19px;
                font-size: 14px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: {'#c92a2a' if active else '#3c3f44'};
                color: {'#ffffff' if active else COLOR_TITLE};
            }}
        """)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # ── Internal state ────────────────────────────────────────────────────
        self._drag_pos        = QPoint()
        self._is_dragging     = False
        self._hovered         = False
        self._recording       = False
        self._question_count  = 0
        self._current_bubble  = None   # streaming answer bubble
        self._typing_widget   = None
        self._typing_row      = None
        self._audio_capture   = None
        self._speech_worker   = None
        self._gemini_worker   = None
        self._context_text    = ""     # candidate context (resume / JD)

        # ── Window setup ─────────────────────────────────────────────────────
        self.setWindowTitle("ClueLy")
        self.setMinimumSize(360, 520)
        self.resize(420, 620)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowOpacity(OPACITY_IDLE)
        self.setMouseTracking(True)

        self._build_ui()

        self._opacity_anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._opacity_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._geo_anim = QPropertyAnimation(self, b"geometry", self)
        self._geo_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self._container = QFrame()
        self._container.setObjectName("container")
        self._container.setStyleSheet(f"""
            QFrame#container {{
                background: {COLOR_BG};
                border: 1px solid {COLOR_BORDER};
                border-radius: 18px;
            }}
        """)

        # Drop shadow on the container
        shadow = QGraphicsDropShadowEffect(self._container)
        shadow.setBlurRadius(32)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 140))
        self._container.setGraphicsEffect(shadow)

        # ── Header ────────────────────────────────────────────────────────────
        logo = QLabel("✦  ClueLy")
        logo.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        logo.setStyleSheet(f"color: {COLOR_TITLE}; background: transparent;")

        self._status_badge = QLabel("  IDLE  ")
        self._status_badge.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self._set_status("IDLE")

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

        ctx_btn = QPushButton("📋")
        ctx_btn.setObjectName("ctxBtn")
        ctx_btn.setFixedSize(28, 26)
        ctx_btn.setToolTip("Set candidate context (resume / JD)")
        ctx_btn.clicked.connect(self._toggle_context_panel)
        ctx_btn.setStyleSheet(f"""
            QPushButton#ctxBtn {{
                background: transparent; color: {COLOR_META};
                border: none; border-radius: 5px; font-size: 14px;
            }}
            QPushButton#ctxBtn:hover {{ background: #3c3f44; color: #ffffff; }}
        """)

        close_btn = QPushButton("×")
        close_btn.setObjectName("closeBtn")
        close_btn.setFixedSize(28, 26)
        close_btn.clicked.connect(self._safe_close)
        close_btn.setStyleSheet(f"""
            QPushButton#closeBtn {{
                background: transparent; color: {COLOR_META};
                border: none; border-radius: 5px; font-size: 18px;
            }}
            QPushButton#closeBtn:hover {{ background: #3c3f44; color: #ff6b6b; }}
        """)

        header = QHBoxLayout()
        header.setContentsMargins(14, 10, 10, 8)
        header.addWidget(logo)
        header.addWidget(self._status_badge)
        header.addStretch()
        header.addWidget(ctx_btn)
        header.addWidget(clear_btn)
        header.addWidget(close_btn)

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"color: {COLOR_BORDER};")

        # ── Context panel (hidden by default) ─────────────────────────────────
        self._ctx_panel = QFrame()
        self._ctx_panel.setVisible(False)
        self._ctx_panel.setStyleSheet(f"""
            QFrame {{
                background: {COLOR_SURFACE};
                border-bottom: 1px solid {COLOR_BORDER};
            }}
        """)
        ctx_hint = QLabel("Paste your resume, skills, or job description:")
        ctx_hint.setFont(QFont("Segoe UI", 8))
        ctx_hint.setStyleSheet(f"color: {COLOR_META}; background: transparent;")

        self._ctx_edit = QTextEdit()
        self._ctx_edit.setPlaceholderText(
            "E.g.  5 years Python, FastAPI, distributed systems…"
        )
        self._ctx_edit.setFont(QFont("Segoe UI", 9))
        self._ctx_edit.setFixedHeight(72)
        self._ctx_edit.setAcceptRichText(False)
        self._ctx_edit.setStyleSheet(f"""
            QTextEdit {{
                background: {COLOR_INPUT_BG}; color: {COLOR_TITLE};
                border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 6px 8px;
            }}
            QTextEdit:focus {{ border: 1px solid {COLOR_ACCENT}; }}
        """)
        self._ctx_edit.textChanged.connect(
            lambda: setattr(self, "_context_text", self._ctx_edit.toPlainText())
        )

        ctx_layout = QVBoxLayout(self._ctx_panel)
        ctx_layout.setContentsMargins(10, 6, 10, 8)
        ctx_layout.setSpacing(4)
        ctx_layout.addWidget(ctx_hint)
        ctx_layout.addWidget(self._ctx_edit)

        # ── Chat area ─────────────────────────────────────────────────────────
        self._msg_container = QWidget()
        self._msg_container.setStyleSheet(f"background: {COLOR_BG};")
        self._msg_layout = QVBoxLayout(self._msg_container)
        self._msg_layout.setContentsMargins(10, 12, 10, 12)
        self._msg_layout.setSpacing(10)
        self._msg_layout.addStretch()

        self._scroll = ChatScrollArea()
        self._scroll.setWidget(self._msg_container)

        # ── Live transcript preview ────────────────────────────────────────────
        self._preview_lbl = QLabel("")
        self._preview_lbl.setFont(QFont("Segoe UI", 9))
        self._preview_lbl.setWordWrap(True)
        self._preview_lbl.setStyleSheet(
            f"color: {COLOR_META}; background: transparent; font-style: italic;"
        )
        self._preview_lbl.setMinimumHeight(18)

        # ── Bottom controls ───────────────────────────────────────────────────
        self._record_btn = RecordButton()
        self._record_btn.clicked.connect(self._toggle_recording)

        self._send_btn = QPushButton("✉ Send")
        self._send_btn.setObjectName("sendBtn")
        self._send_btn.setEnabled(False)
        self._send_btn.setFixedHeight(38)
        self._send_btn.setMinimumWidth(80)
        self._send_btn.clicked.connect(self._manual_send)
        self._send_btn.setStyleSheet(f"""
            QPushButton#sendBtn {{
                background: {COLOR_SEND_BG}; color: #ffffff;
                border: none; border-radius: 10px;
                font-size: 12px; font-weight: 600; padding: 0 14px;
            }}
            QPushButton#sendBtn:hover {{ background: {COLOR_SEND_HV}; }}
            QPushButton#sendBtn:disabled {{
                background: #2c2e33; color: {COLOR_META};
                border: 1px solid {COLOR_BORDER};
            }}
        """)

        device_row = QHBoxLayout()
        device_row.setContentsMargins(12, 0, 12, 0)
        self._device_lbl = QLabel("No audio device")
        self._device_lbl.setFont(QFont("Segoe UI", 8))
        self._device_lbl.setStyleSheet(
            f"color: {COLOR_META}; background: transparent;"
        )
        device_row.addWidget(self._device_lbl)

        controls = QHBoxLayout()
        controls.setContentsMargins(10, 4, 10, 4)
        controls.setSpacing(8)
        controls.addWidget(self._record_btn)
        controls.addWidget(self._preview_lbl, 1)
        controls.addWidget(self._send_btn)

        div2 = QFrame()
        div2.setFrameShape(QFrame.Shape.HLine)
        div2.setStyleSheet(f"color: {COLOR_BORDER};")

        bottom = QVBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(2)
        bottom.addWidget(div2)
        bottom.addLayout(device_row)
        bottom.addLayout(controls)

        # ── Assemble body ─────────────────────────────────────────────────────
        body = QVBoxLayout(self._container)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addLayout(header)
        body.addWidget(div)
        body.addWidget(self._ctx_panel)
        body.addWidget(self._scroll, 1)
        body.addLayout(bottom)
        body.setContentsMargins(0, 0, 0, 8)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.addWidget(self._container)

        # Welcome message
        self._add_bot_bubble(
            "👋 Hi! I'm ClueLy — your live interview assistant.\n\n"
            "Click ▶ to start recording your meeting. I'll listen for each "
            "question, transcribe it, and give you an answer in real-time.\n\n"
            "Optionally paste your resume/JD with 📋 for tailored answers."
        )

    # ── Recording control ─────────────────────────────────────────────────────

    def _toggle_recording(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        try:
            from audio.capture import AudioCapture
            from audio.speech_client import SpeechStreamWorker
        except ImportError as e:
            self._add_bot_bubble(f"⚠ Import error: {e}")
            return

        self._audio_capture = AudioCapture()
        try:
            device_info = self._audio_capture.start()
        except Exception as exc:
            self._add_bot_bubble(f"⚠ Could not open audio device: {exc}")
            return

        self._device_lbl.setText(f"🎙 {device_info}")

        self._speech_worker = SpeechStreamWorker(self._audio_capture)
        self._speech_worker.interim_text.connect(self._on_interim)
        self._speech_worker.question_ready.connect(self._on_question_ready)
        self._speech_worker.error_occurred.connect(self._on_stt_error)
        self._speech_worker.start()

        self._recording = True
        self._record_btn.set_recording(True)
        self._send_btn.setEnabled(True)
        self._set_status("LISTENING")
        self._add_bot_bubble("🔴 Recording started. I'm listening for questions…")

    def _stop_recording(self):
        self._recording = False
        self._record_btn.set_recording(False)
        self._send_btn.setEnabled(False)
        self._set_status("IDLE")
        self._preview_lbl.setText("")

        if self._speech_worker:
            self._speech_worker.stop()
            self._speech_worker.quit()
            self._speech_worker.wait(2000)
            self._speech_worker = None

        if self._audio_capture:
            self._audio_capture.stop()
            self._audio_capture = None

        self._device_lbl.setText("No audio device")
        self._add_bot_bubble("⏹ Recording stopped.")

    # ── Transcript signals ────────────────────────────────────────────────────

    def _on_interim(self, text: str):
        self._preview_lbl.setText(text)

    def _on_question_ready(self, question: str):
        question = question.strip()
        if not question:
            return
        self._preview_lbl.setText("")
        self._question_count += 1

        # Section separator
        sep = SectionLabel(f"─── Q{self._question_count} ───")
        pos = self._msg_layout.count() - 1
        self._msg_layout.insertWidget(pos, sep)

        self._add_user_bubble(question)
        self._start_gemini(question)

    def _on_stt_error(self, error: str):
        self._add_bot_bubble(f"⚠ STT Error: {error}")
        self._set_status("ERR")

    # ── Manual Send ───────────────────────────────────────────────────────────

    def _manual_send(self):
        if self._speech_worker:
            self._speech_worker.flush()

    # ── Gemini streaming ──────────────────────────────────────────────────────

    def _start_gemini(self, question: str):
        if self._gemini_worker and self._gemini_worker.isRunning():
            self._gemini_worker.requestInterruption()
            self._gemini_worker.wait(500)

        self._set_status("THINKING")
        self._show_typing()

        from chain.gemini_worker import GeminiWorker
        self._gemini_worker = GeminiWorker(
            question=question,
            context=self._context_text,
        )
        self._gemini_worker.token_ready.connect(self._on_token)
        self._gemini_worker.answer_done.connect(self._on_answer_done)
        self._gemini_worker.error_occurred.connect(self._on_gemini_error)
        self._gemini_worker.start()

    def _on_token(self, token: str):
        if self._typing_widget:
            self._hide_typing()
            self._current_bubble = self._add_bot_bubble("")
        if self._current_bubble:
            self._current_bubble.append_text(token)
            self._scroll.scroll_to_bottom()

    def _on_answer_done(self):
        self._current_bubble = None
        self._set_status("LISTENING" if self._recording else "IDLE")

    def _on_gemini_error(self, error: str):
        self._hide_typing()
        self._add_bot_bubble(f"⚠ Gemini error: {error}")
        self._current_bubble = None
        self._set_status("ERR")

    # ── Chat helpers ──────────────────────────────────────────────────────────

    def _add_user_bubble(self, text: str) -> MessageBubble:
        bubble = MessageBubble(text, is_user=True)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(bubble)
        pos = self._msg_layout.count() - 1
        self._msg_layout.insertLayout(pos, row)
        self._scroll.scroll_to_bottom()
        return bubble

    def _add_bot_bubble(self, text: str) -> MessageBubble:
        bubble = MessageBubble(text, is_user=False)
        row = QHBoxLayout()
        row.addWidget(bubble)
        row.addStretch()
        pos = self._msg_layout.count() - 1
        self._msg_layout.insertLayout(pos, row)
        self._scroll.scroll_to_bottom()
        return bubble

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
        if self._typing_widget:
            self._typing_widget.stop()
        if self._typing_row:
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
            elif item.widget():
                item.widget().deleteLater()
        self._question_count = 0
        self._add_bot_bubble("Conversation cleared. Ready for new questions.")

    def _toggle_context_panel(self):
        visible = self._ctx_panel.isVisible()
        self._ctx_panel.setVisible(not visible)

    # ── Status badge ──────────────────────────────────────────────────────────

    def _set_status(self, mode: str):
        colours = {
            "IDLE":      ("#2c2e33", COLOR_META,   "  IDLE  "),
            "LISTENING": ("#862e2e", "#fa5252",     "  LIVE  "),
            "THINKING":  ("#1971c2", "#339af0",     "  AI ⟳  "),
            "ERR":       ("#c92a2a", "#ff6b6b",     "  ERR  "),
        }
        bg, fg, label = colours.get(mode, ("#444", "#aaa", f"  {mode}  "))
        self._status_badge.setText(label)
        self._status_badge.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 6px;"
            f" font-size: 8px; font-weight: 700; padding: 2px 5px;"
        )

    # ── Safe close ────────────────────────────────────────────────────────────

    def _safe_close(self):
        if self._recording:
            self._stop_recording()
        self.close()

    # ── Hover / fade animations ───────────────────────────────────────────────

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

    # ── Window dragging ───────────────────────────────────────────────────────

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