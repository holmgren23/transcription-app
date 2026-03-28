import platform
import tempfile
import threading
import wave
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioCapture:
    """Cross-platform system audio recorder using sounddevice."""

    SAMPLE_RATE = 16000
    CHANNELS = 1
    DTYPE = "int16"
    BLOCKSIZE = 1024

    def __init__(self) -> None:
        self.recording = False
        self.frames: list = []
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, device: Optional[int] = None) -> None:
        """Begin recording from *device* (None = default input)."""
        self.recording = True
        self.frames = []
        self._error = None
        self._thread = threading.Thread(
            target=self._record_loop, args=(device,), daemon=True
        )
        self._thread.start()

    def stop(self) -> str:
        """Stop recording and return path to a temporary WAV file."""
        self.recording = False
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
                    self.frames.append(data.copy())
        except Exception as exc:
            self._error = str(exc)

    def _save_to_temp(self) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.SAMPLE_RATE)
            if self.frames:
                wf.writeframes(np.concatenate(self.frames).tobytes())
        return tmp.name
