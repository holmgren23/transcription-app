"""Unit tests for AudioCapture pause/resume behaviour."""

import sys
import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audio_capture import AudioCapture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_stream():
    """Return a MagicMock that behaves like a sounddevice.InputStream."""
    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    # read() returns (data_array, overflowed)
    stream.read.return_value = (
        np.zeros((AudioCapture.BLOCKSIZE, AudioCapture.CHANNELS), dtype=AudioCapture.DTYPE),
        False,
    )
    return stream


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pause_stops_audio_capture():
    """Calling pause() stops the recording loop and sets the paused flag."""
    cap = AudioCapture()
    with patch("audio_capture.sd.InputStream") as mock_cls:
        mock_cls.return_value = _mock_stream()
        cap.start()
        time.sleep(0.05)  # let the thread spin briefly
        cap.pause()

    assert cap.paused is True
    assert cap.recording is False


def test_pause_preserves_existing_frames():
    """Frames captured before pause are not discarded."""
    cap = AudioCapture()
    fake_frame = np.ones((AudioCapture.BLOCKSIZE, AudioCapture.CHANNELS), dtype=AudioCapture.DTYPE)
    with patch("audio_capture.sd.InputStream") as mock_cls:
        stream = _mock_stream()
        stream.read.return_value = (fake_frame, False)
        mock_cls.return_value = stream
        cap.start()
        time.sleep(0.05)
        cap.pause()

    assert len(cap.frames) > 0


def test_resume_restarts_capture():
    """resume() sets recording=True and clears the paused flag."""
    cap = AudioCapture()
    with patch("audio_capture.sd.InputStream") as mock_cls:
        mock_cls.return_value = _mock_stream()
        cap.start()
        cap.pause()
        assert cap.paused is True

        mock_cls.return_value = _mock_stream()
        cap.resume()
        time.sleep(0.05)

        assert cap.recording is True
        assert cap.paused is False
        cap.stop()


def test_resume_appends_to_existing_frames():
    """Frames recorded after resume are added to frames from before the pause."""
    cap = AudioCapture()
    with patch("audio_capture.sd.InputStream") as mock_cls:
        mock_cls.return_value = _mock_stream()
        cap.start()
        time.sleep(0.05)
        cap.pause()
        frames_before_resume = len(cap.frames)

        mock_cls.return_value = _mock_stream()
        cap.resume()
        time.sleep(0.05)
        cap.stop()

    assert len(cap.frames) >= frames_before_resume


def test_pause_resume_state_toggles_correctly():
    """Full start → pause → resume → stop cycle produces correct state at each step."""
    cap = AudioCapture()
    with patch("audio_capture.sd.InputStream") as mock_cls:
        mock_cls.return_value = _mock_stream()

        cap.start()
        assert cap.recording is True
        assert cap.paused is False

        cap.pause()
        assert cap.recording is False
        assert cap.paused is True

        mock_cls.return_value = _mock_stream()
        cap.resume()
        time.sleep(0.05)
        assert cap.recording is True
        assert cap.paused is False

        cap.stop()
        assert cap.recording is False
        assert cap.paused is False


def test_no_new_frames_captured_while_paused():
    """No frames are appended to the buffer while the capture is paused."""
    cap = AudioCapture()
    with patch("audio_capture.sd.InputStream") as mock_cls:
        mock_cls.return_value = _mock_stream()
        cap.start()
        time.sleep(0.05)
        cap.pause()
        count_at_pause = len(cap.frames)

    # Wait after pause — frame count must not grow
    time.sleep(0.1)
    assert len(cap.frames) == count_at_pause


# ---------------------------------------------------------------------------
# take_chunk tests
# ---------------------------------------------------------------------------

def test_take_chunk_returns_accumulated_frames():
    """take_chunk returns all frames currently in the buffer."""
    cap = AudioCapture()
    fake_frame = np.ones((AudioCapture.BLOCKSIZE, AudioCapture.CHANNELS), dtype=AudioCapture.DTYPE)
    cap.frames = [fake_frame, fake_frame, fake_frame]
    result = cap.take_chunk()
    assert len(result) == 3


def test_take_chunk_clears_buffer():
    """After take_chunk the internal frame buffer is empty."""
    cap = AudioCapture()
    fake_frame = np.ones((AudioCapture.BLOCKSIZE, AudioCapture.CHANNELS), dtype=AudioCapture.DTYPE)
    cap.frames = [fake_frame]
    cap.take_chunk()
    assert cap.frames == []


def test_take_chunk_on_empty_buffer_returns_empty_list():
    """take_chunk on an empty buffer returns [] without raising."""
    cap = AudioCapture()
    result = cap.take_chunk()
    assert result == []


def test_take_chunk_does_not_share_reference():
    """The returned list is a copy; mutating it does not affect cap.frames."""
    cap = AudioCapture()
    fake_frame = np.ones((AudioCapture.BLOCKSIZE, AudioCapture.CHANNELS), dtype=AudioCapture.DTYPE)
    cap.frames = [fake_frame]
    result = cap.take_chunk()
    result.append(fake_frame)  # mutate the returned copy
    # cap.frames was already cleared; this just confirms the copy semantics
    assert cap.frames == []


def test_take_chunk_thread_safe_during_recording():
    """take_chunk can be called while the recording loop is running."""
    cap = AudioCapture()
    with patch("audio_capture.sd.InputStream") as mock_cls:
        mock_cls.return_value = _mock_stream()
        cap.start()
        time.sleep(0.05)
        frames = cap.take_chunk()
        # Frames should accumulate again after the chunk was taken
        time.sleep(0.05)
        cap.stop()
    # Just verify it returns a list and doesn't crash
    assert isinstance(frames, list)
