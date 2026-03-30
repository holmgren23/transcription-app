"""Main tkinter UI for the Swedish Lecture Transcription app."""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

# Sibling imports (src/ is already on sys.path via main.py)
from audio_capture import AudioCapture
from chunked_transcription import ChunkedTranscriptionManager
from config import load_api_key, save_api_key
from transcription import transcribe

# ---------------------------------------------------------------------------
# Colours & fonts
# ---------------------------------------------------------------------------
BG = "#1e1e2e"
BG_PANEL = "#2a2a3e"
BG_ENTRY = "#313145"
FG = "#cdd6f4"
FG_DIM = "#6c7086"
ACCENT = "#89b4fa"
SUCCESS = "#a6e3a1"
WARNING = "#f9e2af"
DANGER = "#f38ba8"

FONT_TITLE = ("Segoe UI", 18, "bold")
FONT_LABEL = ("Segoe UI", 10)
FONT_BUTTON = ("Segoe UI", 10, "bold")
FONT_MONO = ("Consolas", 10)


def _configure_styles(root: tk.Tk) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=0)
    style.configure(
        "TNotebook.Tab",
        background=BG_PANEL,
        foreground=FG_DIM,
        padding=[16, 8],
        font=FONT_LABEL,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", BG)],
        foreground=[("selected", ACCENT)],
    )
    style.configure("TFrame", background=BG)
    style.configure("TCombobox", fieldbackground=BG_ENTRY, background=BG_PANEL, foreground=FG)




# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class TranscriptionApp:
    """Swedish lecture transcription desktop application."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.capturer = AudioCapture()
        self._is_recording = False
        self._is_paused = False
        self._audio_file: Optional[str] = None
        self._upload_file: Optional[str] = None
        self._chunk_manager: Optional[ChunkedTranscriptionManager] = None

        self._setup_window()
        _configure_styles(root)
        self._build_ui()
        self._load_saved_key()

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self) -> None:
        self.root.title("Swedish Lecture Transcriber")
        self.root.geometry("820x700")
        self.root.minsize(700, 580)
        self.root.configure(bg=BG)
        try:
            self.root.iconbitmap(default="")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self.root, bg=BG, pady=16)
        header.pack(fill="x", padx=24)
        tk.Label(
            header,
            text="Swedish Lecture Transcriber",
            font=FONT_TITLE,
            bg=BG,
            fg=ACCENT,
        ).pack(side="left")

        # Notebook
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        self._tab_record = ttk.Frame(nb)
        self._tab_upload = ttk.Frame(nb)
        self._tab_settings = ttk.Frame(nb)
        self._tab_help = ttk.Frame(nb)

        nb.add(self._tab_record, text="  Record Audio  ")
        nb.add(self._tab_upload, text="  Upload File  ")
        nb.add(self._tab_settings, text="  Settings  ")
        nb.add(self._tab_help, text="  Help  ")

        self._build_record_tab()
        self._build_upload_tab()
        self._build_settings_tab()
        self._build_help_tab()

        # Transcription output + save (shared across tabs)
        self._build_output_section()

    # ------------------------------------------------------------------
    # Record tab
    # ------------------------------------------------------------------

    def _build_record_tab(self) -> None:
        parent = self._tab_record
        parent.configure(style="TFrame")
        parent.columnconfigure(0, weight=1)

        pad = dict(padx=20, pady=10)

        # Description
        tk.Label(
            parent,
            text=(
                "Play your lecture video in a browser or media player, "
                "then press Record to capture system audio."
            ),
            font=FONT_LABEL,
            bg=BG,
            fg=FG_DIM,
            wraplength=700,
            justify="left",
        ).grid(row=0, column=0, sticky="w", **pad)

        # Device selector
        dev_frame = tk.Frame(parent, bg=BG)
        dev_frame.grid(row=1, column=0, sticky="ew", **pad)

        tk.Label(dev_frame, text="Input device:", font=FONT_LABEL, bg=BG, fg=FG).pack(
            side="left", padx=(0, 8)
        )

        self._device_var = tk.StringVar(value="Default")
        devices = AudioCapture.list_devices()
        device_names = ["Default"] + [d["name"] for d in devices]
        self._device_map = {d["name"]: d["index"] for d in devices}

        self._device_combo = ttk.Combobox(
            dev_frame,
            textvariable=self._device_var,
            values=device_names,
            state="readonly",
            width=40,
        )
        self._device_combo.pack(side="left")

        tk.Button(
            dev_frame,
            text="Refresh",
            font=FONT_LABEL,
            bg=BG_PANEL,
            fg=FG,
            relief="flat",
            command=self._refresh_devices,
        ).pack(side="left", padx=(8, 0))

        # Record button + status
        rec_frame = tk.Frame(parent, bg=BG)
        rec_frame.grid(row=2, column=0, sticky="w", **pad)

        self._rec_btn = tk.Button(
            rec_frame,
            text="⏺  Start Recording",
            font=FONT_BUTTON,
            bg=ACCENT,
            fg=BG,
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
            command=self._toggle_record,
        )
        self._rec_btn.pack(side="left")

        self._pause_btn = tk.Button(
            rec_frame,
            text="⏸  Pause",
            font=FONT_BUTTON,
            bg=BG_PANEL,
            fg=FG,
            relief="flat",
            padx=12,
            pady=8,
            cursor="hand2",
            state="disabled",
            command=self._toggle_pause,
        )
        self._pause_btn.pack(side="left", padx=(8, 0))

        self._rec_status = tk.Label(
            rec_frame,
            text="",
            font=FONT_LABEL,
            bg=BG,
            fg=FG_DIM,
        )
        self._rec_status.pack(side="left", padx=(16, 0))

        # Transcribe recorded audio
        self._rec_transcribe_btn = tk.Button(
            parent,
            text="Transcribe Recording",
            font=FONT_BUTTON,
            bg=SUCCESS,
            fg=BG,
            relief="flat",
            padx=14,
            pady=7,
            cursor="hand2",
            state="disabled",
            command=self._transcribe_recording,
        )
        self._rec_transcribe_btn.grid(row=3, column=0, sticky="w", **pad)

    # ------------------------------------------------------------------
    # Upload tab
    # ------------------------------------------------------------------

    def _build_upload_tab(self) -> None:
        parent = self._tab_upload
        parent.configure(style="TFrame")
        parent.columnconfigure(0, weight=1)

        pad = dict(padx=20, pady=10)

        tk.Label(
            parent,
            text="Upload a local audio or video file for transcription.",
            font=FONT_LABEL,
            bg=BG,
            fg=FG_DIM,
        ).grid(row=0, column=0, sticky="w", **pad)

        # File type hint label
        self._drop_label = tk.Label(
            parent,
            text="Supported: mp3 · mp4 · m4a · wav · webm · ogg · flac · mov · avi · mkv",
            font=FONT_LABEL,
            bg=BG_PANEL,
            fg=FG_DIM,
            relief="flat",
            pady=14,
            padx=20,
        )
        self._drop_label.grid(row=1, column=0, sticky="ew", padx=20, pady=6)

        # Browse button
        browse_frame = tk.Frame(parent, bg=BG)
        browse_frame.grid(row=2, column=0, sticky="w", **pad)

        tk.Button(
            browse_frame,
            text="Browse…",
            font=FONT_BUTTON,
            bg=BG_PANEL,
            fg=FG,
            relief="flat",
            padx=14,
            pady=7,
            cursor="hand2",
            command=self._browse_file,
        ).pack(side="left")

        self._upload_file_label = tk.Label(
            browse_frame,
            text="No file selected",
            font=FONT_LABEL,
            bg=BG,
            fg=FG_DIM,
        )
        self._upload_file_label.pack(side="left", padx=(12, 0))

        # Transcribe button
        self._upload_transcribe_btn = tk.Button(
            parent,
            text="Transcribe File",
            font=FONT_BUTTON,
            bg=SUCCESS,
            fg=BG,
            relief="flat",
            padx=14,
            pady=7,
            cursor="hand2",
            state="disabled",
            command=self._transcribe_upload,
        )
        self._upload_transcribe_btn.grid(row=3, column=0, sticky="w", **pad)

    # ------------------------------------------------------------------
    # Settings tab
    # ------------------------------------------------------------------

    def _build_settings_tab(self) -> None:
        parent = self._tab_settings
        parent.configure(style="TFrame")
        parent.columnconfigure(1, weight=1)

        pad = dict(padx=20, pady=12)

        tk.Label(
            parent,
            text="OpenAI API Key",
            font=FONT_LABEL,
            bg=BG,
            fg=FG,
        ).grid(row=0, column=0, sticky="w", **pad)

        self._api_key_var = tk.StringVar()
        key_entry = tk.Entry(
            parent,
            textvariable=self._api_key_var,
            show="•",
            font=FONT_MONO,
            bg=BG_ENTRY,
            fg=FG,
            insertbackground=FG,
            relief="flat",
            width=52,
        )
        key_entry.grid(row=0, column=1, sticky="ew", **pad)

        # Show/hide toggle
        self._show_key = False
        toggle_btn = tk.Button(
            parent,
            text="Show",
            font=FONT_LABEL,
            bg=BG_PANEL,
            fg=FG,
            relief="flat",
            padx=8,
            command=lambda: self._toggle_key_visibility(key_entry, toggle_btn),
        )
        toggle_btn.grid(row=0, column=2, padx=(0, 20))

        tk.Button(
            parent,
            text="Save API Key",
            font=FONT_BUTTON,
            bg=ACCENT,
            fg=BG,
            relief="flat",
            padx=14,
            pady=7,
            cursor="hand2",
            command=self._save_key,
        ).grid(row=1, column=1, sticky="w", **pad)

        self._key_status = tk.Label(
            parent,
            text="",
            font=FONT_LABEL,
            bg=BG,
            fg=SUCCESS,
        )
        self._key_status.grid(row=2, column=0, columnspan=3, sticky="w", **pad)

        tk.Label(
            parent,
            text=(
                "Your key is stored locally in a .env file in the project folder.\n"
                "It is never sent anywhere except directly to OpenAI."
            ),
            font=FONT_LABEL,
            bg=BG,
            fg=FG_DIM,
            justify="left",
        ).grid(row=3, column=0, columnspan=3, sticky="w", **pad)

    # ------------------------------------------------------------------
    # Help tab
    # ------------------------------------------------------------------

    def _build_help_tab(self) -> None:
        parent = self._tab_help
        parent.configure(style="TFrame")

        text = scrolledtext.ScrolledText(
            parent,
            font=FONT_LABEL,
            bg=BG_PANEL,
            fg=FG,
            relief="flat",
            padx=20,
            pady=16,
            wrap="word",
            state="normal",
        )
        text.pack(fill="both", expand=True, padx=16, pady=12)

        help_content = """QUICK START
───────────
1. Go to the Settings tab and enter your OpenAI API key, then click Save.
2. Choose Record Audio or Upload File.
3. Press Transcribe — your Swedish lecture text will appear below.
4. Click Save Transcription to export as a .txt file.


RECORDING SYSTEM AUDIO — WINDOWS
──────────────────────────────────
Windows allows apps to record the "Stereo Mix" or "What You Hear" source
directly. If you don't see this option in the device list:

  • Right-click the speaker icon → Sound settings → More sound settings
  • Recording tab → right-click blank area → "Show Disabled Devices"
  • Enable "Stereo Mix" and select it in this app before recording.


RECORDING SYSTEM AUDIO — MACOS
────────────────────────────────
macOS does NOT allow apps to capture system audio by default.
You must install a free virtual audio driver called BlackHole:

  1. Download from: https://existential.audio/blackhole/
     (Choose the "2ch" version for most users)

  2. Install the .pkg file and restart your Mac.

  3. Open Audio MIDI Setup (Applications → Utilities → Audio MIDI Setup).

  4. Click the "+" button at the bottom left → "Create Multi-Output Device".

  5. Check both your normal speakers/headphones AND BlackHole 2ch.

  6. Open System Settings → Sound → Output → select "Multi-Output Device".
     (You will still hear audio, and BlackHole captures it simultaneously.)

  7. In this app, select "BlackHole 2ch" from the Input Device dropdown
     before starting a recording.

  8. When done recording, switch your Output back to your normal speakers.


FILE UPLOAD
────────────
Supported formats: mp3, mp4, m4a, wav, webm, ogg, flac, mov, avi, mkv
Maximum file size: 25 MB (OpenAI Whisper API limit)

For longer lectures, consider splitting the video into smaller chunks
using a tool like HandBrake or ffmpeg before uploading.


LANGUAGE
─────────
Transcription is set to Swedish (sv) by default and uses OpenAI's
Whisper model, which handles Swedish lectures with high accuracy.
"""
        text.insert("1.0", help_content)
        text.configure(state="disabled")

    # ------------------------------------------------------------------
    # Output section
    # ------------------------------------------------------------------

    def _build_output_section(self) -> None:
        sep = tk.Frame(self.root, bg=FG_DIM, height=1)
        sep.pack(fill="x", padx=16, pady=(4, 0))

        out_header = tk.Frame(self.root, bg=BG)
        out_header.pack(fill="x", padx=20, pady=(8, 4))

        tk.Label(
            out_header,
            text="Transcription",
            font=("Segoe UI", 12, "bold"),
            bg=BG,
            fg=FG,
        ).pack(side="left")

        self._progress_label = tk.Label(
            out_header,
            text="",
            font=FONT_LABEL,
            bg=BG,
            fg=WARNING,
        )
        self._progress_label.pack(side="left", padx=(16, 0))

        tk.Button(
            out_header,
            text="Clear",
            font=FONT_LABEL,
            bg=BG_PANEL,
            fg=FG_DIM,
            relief="flat",
            padx=8,
            command=self._clear_output,
        ).pack(side="right")

        tk.Button(
            out_header,
            text="Save as .txt",
            font=FONT_BUTTON,
            bg=SUCCESS,
            fg=BG,
            relief="flat",
            padx=12,
            pady=4,
            cursor="hand2",
            command=self._save_transcription,
        ).pack(side="right", padx=(0, 8))

        self._output_text = scrolledtext.ScrolledText(
            self.root,
            font=FONT_MONO,
            bg=BG_PANEL,
            fg=FG,
            relief="flat",
            padx=12,
            pady=8,
            wrap="word",
            height=10,
        )
        self._output_text.pack(fill="both", expand=True, padx=16, pady=(0, 12))

    # ------------------------------------------------------------------
    # Event handlers — recording
    # ------------------------------------------------------------------

    def _refresh_devices(self) -> None:
        devices = AudioCapture.list_devices()
        names = ["Default"] + [d["name"] for d in devices]
        self._device_map = {d["name"]: d["index"] for d in devices}
        self._device_combo["values"] = names

    def _toggle_record(self) -> None:
        if not self._is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        api_key = self._api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning(
                "API Key Missing",
                "Please enter your OpenAI API key in the Settings tab first.",
            )
            return

        selected = self._device_var.get()
        device = None if selected == "Default" else self._device_map.get(selected)

        self._is_recording = True
        self._is_paused = False
        self._rec_btn.configure(text="⏹  Stop Recording", bg=DANGER, fg=BG)
        self._pause_btn.configure(state="normal", text="⏸  Pause", bg=BG_PANEL, fg=FG)
        self._rec_status.configure(text="Recording…", fg=DANGER)
        self._rec_transcribe_btn.configure(state="disabled")

        self.capturer.start(device=device)

        # Start chunked transcription: every 10 minutes send accumulated audio
        # to Whisper and append the result while recording continues.
        self._chunk_manager = ChunkedTranscriptionManager(
            capturer=self.capturer,
            api_key=api_key,
            on_chunk=lambda text: self.root.after(
                0, lambda t=text: self._show_result(t)
            ),
            on_error=lambda msg: self.root.after(
                0, lambda m=msg: self._set_progress(f"Chunk error: {m}")
            ),
            interval=600,
        )
        self._chunk_manager.start()

    def _stop_recording(self) -> None:
        self._is_recording = False
        self._is_paused = False
        self._rec_btn.configure(text="⏺  Start Recording", bg=ACCENT, fg=BG)
        self._pause_btn.configure(state="disabled", text="⏸  Pause", bg=BG_PANEL, fg=FG)
        self._rec_status.configure(text="Processing…", fg=WARNING)

        # Stop the chunk timer so no new periodic chunks fire.
        if self._chunk_manager is not None:
            self._chunk_manager.stop()
            self._chunk_manager = None

        api_key = self._api_key_var.get().strip()

        def finish() -> None:
            try:
                self._audio_file = self.capturer.stop()
            except Exception as exc:
                msg = str(exc)
                self.root.after(0, lambda: self._show_error(msg))
                return

            # Check whether there are remaining frames since the last chunk.
            try:
                has_audio = os.path.getsize(self._audio_file) > 44  # WAV header = 44 bytes
            except OSError:
                has_audio = False

            if has_audio and api_key:
                # Auto-transcribe the final (partial) chunk so nothing is lost.
                self.root.after(
                    0,
                    lambda: self._rec_status.configure(
                        text="Transcribing final chunk…", fg=WARNING
                    ),
                )
                try:
                    result = transcribe(self._audio_file, api_key)
                    self.root.after(0, lambda: self._show_result(result))
                    self.root.after(
                        0,
                        lambda: self._rec_status.configure(
                            text="All chunks transcribed.", fg=SUCCESS
                        ),
                    )
                except Exception as exc:
                    msg = str(exc)
                    self.root.after(0, lambda: self._show_error(msg))
            else:
                self.root.after(
                    0,
                    lambda: self._rec_status.configure(
                        text="Recording saved. Ready to transcribe.", fg=SUCCESS
                    ),
                )
                if has_audio:
                    self.root.after(
                        0, lambda: self._rec_transcribe_btn.configure(state="normal")
                    )

        threading.Thread(target=finish, daemon=True).start()

    def _toggle_pause(self) -> None:
        if not self._is_paused:
            self._pause_recording()
        else:
            self._resume_recording()

    def _pause_recording(self) -> None:
        self._is_paused = True
        self._pause_btn.configure(text="▶  Resume", bg=WARNING, fg=BG)
        self._rec_status.configure(text="Paused", fg=WARNING)
        threading.Thread(target=self.capturer.pause, daemon=True).start()

    def _resume_recording(self) -> None:
        self._is_paused = False
        self._pause_btn.configure(text="⏸  Pause", bg=BG_PANEL, fg=FG)
        self._rec_status.configure(text="Recording…", fg=DANGER)
        selected = self._device_var.get()
        device = None if selected == "Default" else self._device_map.get(selected)
        self.capturer.resume(device=device)

    def _transcribe_recording(self) -> None:
        if not self._audio_file:
            return
        self._run_transcription(self._audio_file)

    # ------------------------------------------------------------------
    # Event handlers — file upload
    # ------------------------------------------------------------------

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select audio or video file",
            filetypes=[
                ("Audio/Video files", "*.mp3 *.mp4 *.m4a *.wav *.webm *.ogg *.flac *.mov *.avi *.mkv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._set_upload_file(path)

    def _set_upload_file(self, path: str) -> None:
        self._upload_file = path
        name = os.path.basename(path)
        self._upload_file_label.configure(text=name, fg=FG)
        self._drop_label.configure(text=f"Selected: {name}", fg=ACCENT)
        self._upload_transcribe_btn.configure(state="normal")

    def _transcribe_upload(self) -> None:
        if not self._upload_file:
            return
        self._run_transcription(self._upload_file)

    # ------------------------------------------------------------------
    # Shared transcription runner
    # ------------------------------------------------------------------

    def _run_transcription(self, file_path: str) -> None:
        api_key = self._api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning(
                "API Key Missing",
                "Please enter your OpenAI API key in the Settings tab first.",
            )
            return

        self._set_progress("Transcribing… please wait")

        def worker() -> None:
            try:
                result = transcribe(file_path, api_key)
                self.root.after(0, lambda: self._show_result(result))
            except Exception as exc:
                self.root.after(0, lambda: self._show_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Event handlers — output
    # ------------------------------------------------------------------

    def _show_result(self, text: str) -> None:
        self._set_progress("")
        existing = self._output_text.get("1.0", "end").strip()
        if existing:
            self._output_text.insert("end", "\n\n" + text)
        else:
            self._output_text.insert("end", text)

    def _show_error(self, message: str) -> None:
        self._set_progress("")
        messagebox.showerror("Transcription Error", message)

    def _clear_output(self) -> None:
        self._output_text.delete("1.0", "end")

    def _save_transcription(self) -> None:
        content = self._output_text.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo("Nothing to save", "The transcription is empty.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save transcription",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Saved", f"Transcription saved to:\n{path}")

    # ------------------------------------------------------------------
    # Settings helpers
    # ------------------------------------------------------------------

    def _load_saved_key(self) -> None:
        key = load_api_key()
        if key:
            self._api_key_var.set(key)

    def _save_key(self) -> None:
        key = self._api_key_var.get().strip()
        if not key:
            messagebox.showwarning("Empty Key", "Please enter an API key before saving.")
            return
        save_api_key(key)
        self._key_status.configure(text="✓ API key saved to .env")

    def _toggle_key_visibility(self, entry: tk.Entry, btn: tk.Button) -> None:
        self._show_key = not self._show_key
        entry.configure(show="" if self._show_key else "•")
        btn.configure(text="Hide" if self._show_key else "Show")

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _set_progress(self, msg: str) -> None:
        self._progress_label.configure(text=msg)
        self.root.update_idletasks()
