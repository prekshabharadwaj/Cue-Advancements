# ClueLy – Rebuild Plan: Live Meeting Transcription + AI Answers

## What we're building

A floating, always-on-top PySide6 desktop app that:
1. Records **system audio** (loopback — captures whatever plays through speakers: Zoom, Meet, Teams, etc.)
2. Transcribes speech live using **Google Cloud Speech-to-Text (Streaming API)**
3. Detects each interviewer question via **silence/pause detection** or a **"Send" button**
4. Sends the transcribed question to **Google Gemini API**
5. Streams the answer back as a **new bot bubble** in the chat UI

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    MainWindow (UI)                   │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  Chat area │  │ Record panel │  │ Status bar  │  │
│  │ (bubbles)  │  │ ▶/■  Send   │  │ 🔴 Live     │  │
│  └────────────┘  └──────────────┘  └─────────────┘  │
└──────────────────────────────────────────────────────┘
         │                    │
   GeminiWorker        AudioCaptureThread
   (QThread)           (QThread)
         │                    │
   Gemini API          SpeechBuffer
   (google-generativeai)  ↓
                      Google STT
                      Streaming API
```

---

## New File Structure

```
ClueLy/
├── app.py                     (unchanged entry point)
├── .env                       (add GOOGLE_API_KEY, GOOGLE_APPLICATION_CREDENTIALS)
├── requirements.txt           (updated)
│
├── audio/
│   ├── __init__.py
│   ├── capture.py             ← loopback audio capture (soundcard / pyaudiowpatch)
│   └── speech_client.py      ← Google STT streaming client
│
├── chain/
│   ├── gemini_worker.py       ← QThread worker: sends transcript → Gemini, streams answer
│   └── (old files removed or kept for reference)
│
└── ui/
    ├── main_window.py         ← complete rewrite
    ├── bubbles.py             ← MessageBubble, TypingIndicator (extracted)
    └── styles.py              ← design tokens
```

---

## Implementation Steps

### Step 1 — Dependencies

Update `requirements.txt`:
- Remove: `faiss-cpu`, `pypdf`, `docx2txt`, `unstructured`, `langchain*`
- Add:
  - `google-generativeai` — Gemini API
  - `google-cloud-speech` — Google STT streaming
  - `soundcard` — cross-platform loopback/system audio capture
  - `pyaudio` — raw audio stream fallback
  - `webrtcvad` — Voice Activity Detection (silence/pause detection)
  - `numpy` — audio buffer processing

### Step 2 — Audio Capture (`audio/capture.py`)

- Use `soundcard` to list and open the **default loopback device** (what's playing through speakers)
- Record in 16kHz mono PCM chunks (required by Google STT)
- Emit audio frames to a shared queue

### Step 3 — Speech-to-Text (`audio/speech_client.py`)

- Open a **Google STT Streaming session** on the audio queue
- Use `language_code="en-US"`, `interim_results=True`
- On each `is_final=True` result → emit `question_ready(text)` signal

### Step 4 — Silence / Pause Detection

- Use `webrtcvad` (VAD) to detect speech vs. silence frames
- After a configurable pause (e.g. **1.5 seconds of silence** after speech) → treat the accumulated transcript as one complete question and emit it
- This handles the auto-send case. The manual **"Send"** button flushes the current buffer immediately.

### Step 5 — Gemini Worker (`chain/gemini_worker.py`)

```python
class GeminiWorker(QThread):
    token_ready  = Signal(str)   # each streamed token
    answer_done  = Signal()
    error        = Signal(str)

    def run(self):
        model = genai.GenerativeModel("gemini-1.5-flash")
        for chunk in model.generate_content(self.prompt, stream=True):
            self.token_ready.emit(chunk.text)
        self.answer_done.emit()
```

- `GOOGLE_API_KEY` loaded from `.env`
- Each new question spawns a new `GeminiWorker` thread

### Step 6 — UI Rewrite (`ui/main_window.py`)

Key UI elements:
| Element | Purpose |
|---|---|
| **Chat scroll area** | Shows question (user bubble) + answer (bot bubble) pairs |
| **🔴 / ⏹ Record button** | Starts / stops loopback audio capture |
| **Send button** | Manually flush current transcript buffer as a question |
| **Live transcript preview** | Small italic label showing interim STT text in real-time |
| **Status badge** | Shows IDLE / LISTENING / THINKING |

Behavior flow:
1. User clicks **▶ Record** → `AudioCaptureThread` starts
2. Interim STT text appears in the **live preview** label
3. On silence pause OR **Send** click → question bubble added to chat → `GeminiWorker` starts
4. Answer streams into a bot bubble token-by-token
5. User clicks **⏹ Stop** → recording ends; any buffered text is flushed as a final question

### Step 7 — `.env` updates

```
GOOGLE_API_KEY=your_gemini_api_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json
```

---

## UX Design

- **Opacity behavior**: idle 12% → hover 94% (kept from current design)
- **Always-on-top, frameless** window (kept)
- **Color palette**: unchanged dark theme (`#1a1b1e` base)
- **New record button**: glowing red pulsing dot when active
- **Live preview text**: faint italic text below the input area, updates in real-time
- **Streaming answer**: bot bubble grows token-by-token (text appended live)

---

## Open Questions / Decisions Needed

> [!IMPORTANT]
> The following need to be confirmed before implementation:

1. **Google credentials setup**: The user will need a Google Cloud project with Speech-to-Text API enabled and a service account JSON. Gemini API just needs `GOOGLE_API_KEY`.
2. **Loopback device on Windows**: `soundcard` should auto-detect the default loopback (WASAPI loopback). We should test this works on the user's system.
3. **Silence threshold**: Default 1.5s pause. Should be user-configurable via a settings slider?
4. **Interview context**: Should there be a system prompt field where the user can paste their resume / job description to give Gemini context for better answers?

---

## Implementation Order

1. `requirements.txt` — update deps  
2. `audio/capture.py` — loopback recording  
3. `audio/speech_client.py` — Google STT streaming  
4. `chain/gemini_worker.py` — Gemini streaming worker  
5. `ui/bubbles.py` — extract bubble components  
6. `ui/main_window.py` — full UI rewrite  
7. `.env.example` — update keys  
8. Test end-to-end
