# Transcription App — Project Progress

## What Has Been Built

A cross-platform desktop app that records system audio (or accepts uploaded files) and transcribes speech using the OpenAI Whisper API. Designed for long Swedish university lectures but supports all Whisper languages.

### Core Features
- **Live audio recording** via sounddevice (system audio / microphone)
- **Pause / Resume** recording — frames are preserved across pauses
- **File upload** transcription (mp3, mp4, m4a, wav, webm, ogg, flac, mov, avi, mkv; ≤ 25 MB)
- **Chunked transcription** — every 10 minutes of recording is sent to Whisper automatically while capture continues; text is appended in real-time, making 1–2 hour lectures reliable
- **Automatic final-chunk transcription** — when recording stops, any remaining audio since the last chunk is transcribed and appended automatically
- **API key management** — stored in `.env` via python-dotenv, never hard-coded
- **Dark-themed tkinter UI** — tabbed interface (Record / Upload / Settings / Help)
- **Save transcription** to `.txt` file
- Windows installer built via PyInstaller (see `TranscriptionApp.spec`)

---

## File Map

| File | Purpose |
|------|---------|
| `src/main.py` | Entry point — creates `tk.Tk` root and launches `TranscriptionApp` |
| `src/ui.py` | Full tkinter UI: tabs, recording controls, output area, event handlers |
| `src/audio_capture.py` | `AudioCapture` — sounddevice wrapper; start/pause/resume/stop; `take_chunk()` for chunked transcription |
| `src/chunked_transcription.py` | `ChunkedTranscriptionManager` — 10-minute rolling chunk timer; saves WAV fragments and calls Whisper in parallel threads |
| `src/transcription.py` | `transcribe(file_path, api_key, language)` — thin wrapper around `openai.audio.transcriptions.create` |
| `src/config.py` | `load_api_key()` / `save_api_key()` — reads and writes `.env` |
| `tests/test_audio_capture.py` | Unit tests for AudioCapture (pause/resume state, take_chunk thread safety) |
| `tests/test_chunked_transcription.py` | Unit tests for ChunkedTranscriptionManager (timer, callbacks, flush, error handling) |
| `tests/test_ui.py` | Unit tests for UI state machine (button states, transcription append) |
| `requirements.txt` | Runtime dependencies: openai, python-dotenv, sounddevice, numpy |
| `.env.example` | Template showing required env var (`OPENAI_API_KEY`) |
| `docs/README.md` | User-facing documentation |
| `.github/workflows/build.yml` | CI: builds Windows `.exe` via PyInstaller on push to master |
| `TranscriptionApp.spec` | PyInstaller spec file for Windows executable |

---

## How Chunked Transcription Works

```
Recording thread                    Chunk timer (every 10 min)
     |                                        |
     | append frames → AudioCapture.frames   |
     |                                        | _process_chunk()
     |                                        |   take_chunk()  ← atomically drain frames
     |                                        |   save to temp WAV
     |                                        |   transcribe(WAV)  ← background thread
     |                                        |   on_chunk(text)   ← root.after → _show_result
     |                                        | _schedule_next()
     | ...recording continues...             |
     |
Stop pressed
     | stop chunk manager (cancel timer)
     | capturer.stop() → save remaining frames
     | transcribe remaining WAV automatically
     | _show_result(final text)
```

The `AudioCapture._frames_lock` mutex ensures `take_chunk()` and the recording append loop don't race.

---

## Next Planned Features

- [ ] Language selector in UI (currently hard-coded to Swedish `sv`)
- [ ] Configurable chunk interval (5 / 10 / 15 min slider in Settings)
- [ ] Speaker diarisation (identify who is speaking)
- [ ] Export to `.srt` subtitle format with timestamps
- [ ] macOS / Linux packaging (currently only Windows `.exe` via PyInstaller CI)
- [ ] Drag-and-drop file upload
- [ ] In-app audio level meter (VU meter) during recording
