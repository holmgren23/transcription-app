import platform
import tempfile
import threading
import time
import wave
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


class AudioCapture:
    """Cross-platform system audio recorder using sounddevice."""

    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = "int16"
    BLOCKSIZE = 1024

    # RMS amplitude below which a block is considered silent (int16 scale 0–32767)
    SILENCE_THRESHOLD: int = 100
    # Seconds of continuous silence before the on_silence callback fires
    SILENCE_TIMEOUT: float = 60.0

    def __init__(self) -> None:
        self.recording = False
        self.paused = False
        self.frames: list = []
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None
        self._device: Optional[int] = None
        self._frames_lock = threading.Lock()
        self._on_silence: Optional[Callable[[], None]] = None
        # Allow tests / callers to override these per-instance
        self.silence_threshold: int = self.SILENCE_THRESHOLD
        self.silence_timeout: float = self.SILENCE_TIMEOUT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        device: Optional[int] = None,
        on_silence: Optional[Callable[[], None]] = None,
    ) -> None:
        """Begin recording from *device* (None = default input).

        Parameters
        ----------
        device:
            sounddevice device index, or None for the system default.
        on_silence:
            Optional callable invoked (from the recording thread) when no audio
            above :attr:`silence_threshold` is detected for :attr:`silence_timeout`
            seconds.  The callback should be thread-safe (e.g. schedule work via
            ``root.after``) and must not call :meth:`stop` directly, as that
            would deadlock.
        """
        self.recording = True
        self.paused = False
        self.frames = []
        self._error = None
        self._device = device
        self._on_silence = on_silence
        self._thread = threading.Thread(
            target=self._record_loop, args=(device,), daemon=True
        )
        self._thread.start()

    def pause(self) -> None:
        """Pause capture — stops the stream but keeps all frames so far."""
        self.recording = False
        self.paused = True
        if self._thread:
            self._thread.join(timeout=3)

    def resume(self, device: Optional[int] = None) -> None:
        """Resume capture after a pause, appending to existing frames."""
        if device is None:
            device = self._device
        self.recording = True
        self.paused = False
        self._thread = threading.Thread(
            target=self._record_loop, args=(device,), daemon=True
        )
        self._thread.start()

    def stop(self) -> str:
        """Stop recording and return path to a temporary WAV file."""
        self.recording = False
        self.paused = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._error:
            raise RuntimeError(f"Recording failed: {self._error}")
        return self._save_to_temp()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_devices() -> list:
        """Return a list of available input audio devices."""
        devices = sd.query_devices()
        result = []
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                result.append({"index": idx, "name": dev["name"]})
        return result

    @staticmethod
    def is_macos() -> bool:
        return platform.system() == "Darwin"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_loop(self, device: Optional[int]) -> None:
        on_silence = self._on_silence
        last_active = time.monotonic()
        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype=self.DTYPE,
                device=device,
                blocksize=self.BLOCKSIZE,
            ) as stream:
                while self.recording:
                    data, _ = stream.read(self.BLOCKSIZE)
                    with self._frames_lock:
                        self.frames.append(data.copy())

                    # Silence detection
                    rms = float(np.sqrt(np.mean(data.astype(np.float32) ** 2)))
                    if rms > self.silence_threshold:
                        last_active = time.monotonic()
                    elif on_silence and (time.monotonic() - last_active) > self.silence_timeout:
                        on_silence()
                        break
        except Exception as exc:
            self._error = str(exc)

    def take_chunk(self) -> list:
        """Atomically return current frames and clear the buffer.

        Called by :class:`~chunked_transcription.ChunkedTranscriptionManager`
        to extract audio accumulated since the previous chunk without dropping
        any frames that arrive concurrently from the recording thread.
        """
        with self._frames_lock:
            frames = self.frames[:]
            self.frames = []
        return frames

    def _save_to_temp(self) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.SAMPLE_RATE)
            with self._frames_lock:
                frames = self.frames[:]
            if frames:
                wf.writeframes(np.concatenate(frames).tobytes())
        return tmp.name
