"""
ClueLy – audio/capture.py
Captures system audio (loopback) using soundcard (WASAPI loopback on Windows).
Emits raw 16-bit PCM frames at 16 kHz mono into a thread-safe queue,
ready for Google Speech-to-Text streaming.

Falls back to PyAudio microphone capture if no loopback device is found.
"""

import queue
import threading
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 16_000   # Hz  – required by Google STT
CHANNELS      = 1        # mono
FRAME_MS      = 30       # ms per VAD frame (10 / 20 / 30 allowed by webrtcvad)
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)   # 480 samples
FRAME_BYTES   = FRAME_SAMPLES * 2                    # 16-bit PCM → 2 bytes/sample


class AudioCapture:
    """
    Thread-safe system audio loopback capture.

    Usage
    -----
    cap = AudioCapture()
    cap.start()
    # read frames from cap.queue (bytes objects, FRAME_BYTES each)
    cap.stop()
    """

    def __init__(self):
        self.queue: queue.Queue[bytes | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._device_name: str | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> str:
        """
        Start capturing. Returns a description of the device being used.
        Raises RuntimeError if no suitable device is found.
        """
        self._stop_event.clear()
        # flush stale data
        while not self.queue.empty():
            self.queue.get_nowait()

        backend, device_info = self._pick_device()
        self._device_name = device_info
        if backend == "soundcard":
            self._thread = threading.Thread(
                target=self._record_soundcard, daemon=True
            )
        else:
            self._thread = threading.Thread(
                target=self._record_pyaudio, daemon=True
            )
        self._thread.start()
        return device_info

    def stop(self):
        """Signal the recording thread to stop and drain the queue."""
        self._stop_event.set()
        self.queue.put(None)  # sentinel
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    @property
    def device_name(self) -> str | None:
        return self._device_name

    # ── Device selection ─────────────────────────────────────────────────────

    @staticmethod
    def _pick_device() -> tuple[str, str]:
        """Return ('soundcard', info) or ('pyaudio', info)."""
        try:
            import soundcard as sc
            # Prefer loopback of the default output (WASAPI loopback on Windows)
            try:
                mic = sc.get_microphone(
                    sc.default_speaker().name, include_loopback=True
                )
                return "soundcard", f"Loopback: {mic.name}"
            except Exception:
                pass
            # Fallback: first available loopback device
            for m in sc.all_microphones(include_loopback=True):
                if "loopback" in m.name.lower() or "stereo mix" in m.name.lower():
                    return "soundcard", f"Loopback: {m.name}"
            # Last resort: default microphone via soundcard
            mic = sc.default_microphone()
            return "soundcard", f"Microphone: {mic.name}"
        except ImportError:
            pass

        # soundcard not available → try PyAudio microphone
        return "pyaudio", "Microphone (PyAudio)"

    # ── soundcard recording thread ────────────────────────────────────────────

    def _record_soundcard(self):
        import soundcard as sc

        # Pick the loopback device (same logic as _pick_device)
        try:
            mic = sc.get_microphone(
                sc.default_speaker().name, include_loopback=True
            )
        except Exception:
            mic = sc.default_microphone()

        block_samples = FRAME_SAMPLES  # one VAD frame per read call

        with mic.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS,
                          blocksize=block_samples) as recorder:
            while not self._stop_event.is_set():
                # shape: (block_samples, channels) float32
                data = recorder.record(numframes=block_samples)
                # Convert float32 → int16 PCM
                pcm = (np.clip(data[:, 0], -1.0, 1.0) * 32767).astype(np.int16)
                self.queue.put(pcm.tobytes())

        self.queue.put(None)  # sentinel

    # ── PyAudio fallback recording thread ─────────────────────────────────────

    def _record_pyaudio(self):
        import pyaudio

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SAMPLES,
        )
        try:
            while not self._stop_event.is_set():
                raw = stream.read(FRAME_SAMPLES, exception_on_overflow=False)
                self.queue.put(raw)
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            self.queue.put(None)  # sentinel
