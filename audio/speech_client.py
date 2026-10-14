"""
ClueLy – audio/speech_client.py
Google Cloud Speech-to-Text streaming transcription with
energy-based silence/pause detection (pure Python + numpy — no C compiler needed).

Runs in a QThread. Emits:
  • interim_text(str)   – live partial transcript (shown in preview label)
  • question_ready(str) – final transcript after a pause (triggers LLM)
  • error_occurred(str) – on any unrecoverable error
"""

import queue
import time
import struct

import numpy as np
from PySide6.QtCore import QThread, Signal

from audio.capture import AudioCapture, SAMPLE_RATE, FRAME_SAMPLES, FRAME_BYTES, FRAME_MS

# ── Tuning constants ──────────────────────────────────────────────────────────
SILENCE_TIMEOUT_SEC  = 1.8     # seconds of silence after speech → question done
MIN_SPEECH_MS        = 400     # ignore blips shorter than this (ms)
ENERGY_THRESHOLD     = 300     # RMS threshold for speech vs. silence (0–32767 scale)
                               # lower = more sensitive, raise if noisy environment
LANGUAGE_CODE        = "en-US"


def _rms(frame_bytes: bytes) -> float:
    """Compute RMS energy of a 16-bit PCM frame."""
    count = len(frame_bytes) // 2
    shorts = struct.unpack(f"<{count}h", frame_bytes)
    arr = np.array(shorts, dtype=np.float32)
    return float(np.sqrt(np.mean(arr ** 2)))


class SpeechStreamWorker(QThread):
    """
    Consumes frames from AudioCapture.queue, runs energy-based VAD,
    accumulates speech buffers, sends them to Google STT when a pause
    is detected (or when flush() is called manually).
    """

    interim_text   = Signal(str)   # live partial transcript / status
    question_ready = Signal(str)   # finalized question (after pause)
    error_occurred = Signal(str)

    def __init__(self, audio_capture: AudioCapture, parent=None):
        super().__init__(parent)
        self._capture   = audio_capture
        self._running   = False
        self._flush_now = False   # set True by the manual Send button

    # ── Public API ────────────────────────────────────────────────────────────

    def flush(self):
        """Manually flush the current buffer as a question (Send button)."""
        self._flush_now = True

    def stop(self):
        self._running = False

    # ── Main thread loop ──────────────────────────────────────────────────────

    def run(self):
        self._running = True
        try:
            self._stream_loop()
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))

    def _stream_loop(self):
        try:
            from google.cloud import speech
        except ImportError as e:
            self.error_occurred.emit(
                f"Missing dependency: {e}\n"
                "Run: pip install google-cloud-speech"
            )
            return

        client = speech.SpeechClient()
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=LANGUAGE_CODE,
            enable_automatic_punctuation=True,
            model="latest_long",
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
        )

        # Accumulated state
        speech_buffer:  list[bytes] = []   # raw PCM frames while speaking
        silence_start:  float | None = None
        speech_ms:      int = 0
        frame_count:    int = 0

        def _emit_question():
            nonlocal speech_buffer, silence_start, speech_ms, frame_count
            text = self._transcribe(client, streaming_config, speech_buffer)
            if text:
                self.question_ready.emit(text)
            speech_buffer = []
            silence_start = None
            speech_ms     = 0
            frame_count   = 0
            self.interim_text.emit("")

        # ── Frame loop ────────────────────────────────────────────────────────
        while self._running:

            # Manual flush (Send button)
            if self._flush_now:
                self._flush_now = False
                if speech_buffer:
                    _emit_question()
                continue

            # Check silence timeout even when queue is momentarily empty
            if silence_start is not None and speech_ms >= MIN_SPEECH_MS:
                if (time.monotonic() - silence_start) >= SILENCE_TIMEOUT_SEC:
                    _emit_question()
                    continue

            try:
                frame = self._capture.queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if frame is None:   # sentinel – capture stopped
                if speech_buffer:
                    _emit_question()
                break

            # Skip malformed frames
            if len(frame) != FRAME_BYTES:
                continue

            is_speech = _rms(frame) > ENERGY_THRESHOLD

            if is_speech:
                speech_buffer.append(frame)
                speech_ms    += FRAME_MS
                silence_start = None
                frame_count  += 1

                # Update interim label every ~10 frames
                if frame_count % 10 == 0 and speech_ms >= MIN_SPEECH_MS:
                    self.interim_text.emit("🎙 Listening…")
            else:
                if speech_buffer:
                    if silence_start is None:
                        silence_start = time.monotonic()

    def _transcribe(
        self,
        client,
        streaming_config,
        frames: list[bytes],
    ) -> str:
        """Send buffered PCM to Google STT and return the final transcript."""
        if not frames:
            return ""

        self.interim_text.emit("⏳ Transcribing…")
        audio_data = b"".join(frames)

        # Google STT Streaming: chunk into ≤ 25 600 byte requests (100 ms each)
        CHUNK = 25_600

        def _request_gen():
            for i in range(0, len(audio_data), CHUNK):
                yield __import__("google.cloud.speech", fromlist=["speech"]).StreamingRecognizeRequest(
                    audio_content=audio_data[i: i + CHUNK]
                )

        # Re-import here to avoid circular reference issues inside generator
        from google.cloud import speech as gcs

        def _request_gen2():
            for i in range(0, len(audio_data), CHUNK):
                yield gcs.StreamingRecognizeRequest(
                    audio_content=audio_data[i: i + CHUNK]
                )

        try:
            responses = client.streaming_recognize(streaming_config, _request_gen2())
            parts = []
            for resp in responses:
                for result in resp.results:
                    if result.is_final:
                        parts.append(result.alternatives[0].transcript.strip())
            return " ".join(parts)
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(f"STT error: {exc}")
            return ""
