"""Chunked transcription manager for long recordings.

Periodically snapshots the audio buffer, saves it to a temp WAV file, and
sends it to the Whisper API while recording continues in parallel.
"""

import os
import tempfile
import threading
import wave
from typing import Callable, Optional

import numpy as np

from audio_capture import AudioCapture
from transcription import transcribe


class ChunkedTranscriptionManager:
    """Transcribes audio in rolling chunks during a live recording.

    Every *interval* seconds the manager calls :meth:`AudioCapture.take_chunk`
    to atomically grab accumulated frames and clear the buffer, writes them to
    a temporary WAV file, and dispatches a background thread to call the
    Whisper API.  Results arrive via *on_chunk*; errors via *on_error*.

    Parameters
    ----------
    capturer : AudioCapture
        The live recording object whose frames will be chunked.
    api_key : str
        OpenAI API key forwarded to :func:`~transcription.transcribe`.
    on_chunk : Callable[[str], None]
        Called with the transcribed text when a chunk completes.
    on_error : Callable[[str], None], optional
        Called with an error message string on transcription failure.
    interval : int
        Seconds between chunks.  Defaults to 600 (10 minutes).
    language : str
        BCP-47 language code passed to Whisper.  Defaults to "sv".
    """

    def __init__(
        self,
        capturer: AudioCapture,
        api_key: str,
        on_chunk: Callable[[str], None],
        on_error: Optional[Callable[[str], None]] = None,
        interval: int = 600,
        language: str = "sv",
    ) -> None:
        self._capturer = capturer
        self._api_key = api_key
        self._on_chunk = on_chunk
        self._on_error = on_error
        self._interval = interval
        self._language = language
        self._timer: Optional[threading.Timer] = None
        self._active = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the periodic chunk timer."""
        self._active = True
        self._schedule_next()

    def stop(self) -> None:
        """Cancel the pending timer.  Any in-flight transcription threads
        continue to completion and will still call *on_chunk*."""
        self._active = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def flush(self) -> None:
        """Immediately take any remaining frames and transcribe them.

        Intended to be called when recording stops so that the tail end of
        the audio is not lost.  Runs asynchronously — *on_chunk* fires once
        the API call returns.
        """
        frames = self._capturer.take_chunk()
        if frames:
            threading.Thread(
                target=self._transcribe_frames, args=(frames,), daemon=True
            ).start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _schedule_next(self) -> None:
        self._timer = threading.Timer(self._interval, self._process_chunk)
        self._timer.daemon = True
        self._timer.start()

    def _process_chunk(self) -> None:
        if not self._active:
            return
        frames = self._capturer.take_chunk()
        if frames:
            threading.Thread(
                target=self._transcribe_frames, args=(frames,), daemon=True
            ).start()
        # Always schedule the next chunk regardless of whether this one had audio
        self._schedule_next()

    def _transcribe_frames(self, frames: list) -> None:
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(AudioCapture.CHANNELS)
                wf.setsampwidth(2)  # 16-bit = 2 bytes per sample
                wf.setframerate(AudioCapture.SAMPLE_RATE)
                wf.writeframes(np.concatenate(frames).tobytes())

            text = transcribe(tmp_path, self._api_key, self._language)
            self._on_chunk(text)
        except Exception as exc:
            if self._on_error is not None:
                self._on_error(str(exc))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
