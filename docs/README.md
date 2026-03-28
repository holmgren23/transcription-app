# Swedish Lecture Transcriber

A cross-platform desktop app for transcribing Swedish video lectures using OpenAI Whisper API.

## Features

- **Real-time system audio capture** — play a lecture in any browser or media player while the app listens
- **File upload** — drag & drop or browse for a local audio/video file
- **Swedish language** — transcription optimised for Swedish via Whisper
- **Save output** — export transcription as a `.txt` file
- **Secure key storage** — API key saved locally in `.env`, never hardcoded

## Requirements

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- **macOS only**: [BlackHole](https://existential.audio/blackhole/) virtual audio driver for system audio capture

---

## Setup

### 1. Clone / download the project

```bash
cd transcription-app
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note — drag & drop**: `tkinterdnd2` requires Tcl/Tk 8.6+. If installation fails on your platform, drag-and-drop will be disabled but the Browse button still works.

> **Note — macOS audio**: On macOS you need to install PortAudio first:
> ```bash
> brew install portaudio
> ```

### 4. Configure your API key

Either:
- Copy `.env.example` to `.env` and paste your key, **or**
- Open the app and enter your key in the **Settings** tab, then click Save

---

## Running the app

```bash
python src/main.py
```

---

## macOS — System Audio Setup

macOS blocks apps from recording system audio by default.
Install **BlackHole** (free, open source) to route audio through a virtual device:

1. Download from **https://existential.audio/blackhole/** (choose "2ch")
2. Run the installer and restart your Mac
3. Open **Audio MIDI Setup** (`/Applications/Utilities/Audio MIDI Setup`)
4. Click **+** → **Create Multi-Output Device**
5. Tick both your **speakers/headphones** and **BlackHole 2ch**
6. Go to **System Settings → Sound → Output** → select **Multi-Output Device**
   *(You will still hear audio; BlackHole silently captures it too)*
7. In the app, select **BlackHole 2ch** as the input device
8. After recording, switch your system output back to your normal speakers

Full instructions are also available in the app's **Help** tab.

---

## Windows — System Audio Setup

1. Right-click the speaker icon in the taskbar → **Sound settings → More sound settings**
2. Go to the **Recording** tab
3. Right-click blank space → **Show Disabled Devices**
4. Enable **Stereo Mix** (or "What U Hear")
5. Select it from the device dropdown in the app before recording

---

## Supported File Formats

`mp3` · `mp4` · `m4a` · `wav` · `webm` · `ogg` · `flac` · `mov` · `avi` · `mkv`

> Maximum file size: **25 MB** (OpenAI Whisper API limit).
> For longer files, split them first with [HandBrake](https://handbrake.fr/) or `ffmpeg`.

---

## Project Structure

```
transcription-app/
├── src/
│   ├── main.py            # Entry point
│   ├── ui.py              # tkinter interface
│   ├── audio_capture.py   # Sounddevice recording
│   ├── transcription.py   # OpenAI Whisper API
│   └── config.py          # .env API key management
├── docs/
│   └── README.md
├── .env.example
└── requirements.txt
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `sounddevice` install fails on macOS | Run `brew install portaudio` first |
| No audio devices shown | Click **Refresh** next to the device dropdown |
| `ModuleNotFoundError: tkinterdnd2` | Drag-and-drop disabled; use the Browse button |
| File too large error | Keep files under 25 MB; split large videos |
| Transcription in wrong language | Language is fixed to Swedish (`sv`) by default |

---

## License

MIT
