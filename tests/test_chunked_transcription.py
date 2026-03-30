"""Unit tests for ChunkedTranscriptionManager."""

import os
import sys
import time
import threading
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from audio_capture import AudioCapture
from chunked_transcription import ChunkedTranscriptionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frames(n_blocks: int = 10) -> list:
    """Return a list of audio frame arrays (non-silent so WAV is non-empty)."""
    return [
        np.ones((AudioCapture.BLOCKSIZE, AudioCapture.CHANNELS), dtype=AudioCapture.DTYPE)
        for _ in range(n_blocks)
    ]


@pytest.fixture()
def capturer():
    """AudioCapture mock that returns real-ish frames from take_chunk."""
    cap = MagicMock(spec=AudioCapture)
    cap.take_chunk.return_value = _make_frames()
    cap.CHANNELS = AudioCapture.CHANNELS
    cap.SAMPLE_RATE = AudioCapture.SAMPLE_RATE
    cap.BLOCKSIZE = AudioCapture.BLOCKSIZE
    cap.DTYPE = AudioCapture.DTYPE
    return cap


# ---------------------------------------------------------------------------
# Timer scheduling
# ---------------------------------------------------------------------------

def test_start_schedules_timer_with_correct_interval(capturer):
    """start() creates a Timer for the configured interval."""
    on_chunk = MagicMock()
    manager = ChunkedTranscriptionManager(capturer, "sk-test", on_chunk, interval=300)
    with patch("chunked_transcription.threading.Timer") as mock_timer_cls:
        mock_timer = MagicMock()
        mock_timer_cls.return_value = mock_timer
        manager.start()
        mock_timer_cls.assert_called_once_with(300, manager._process_chunk)
        mock_timer.start.assert_called_once()
    manager.stop()


def test_stop_cancels_timer(capturer):
    """stop() cancels the pending timer and clears _active."""
    on_chunk = MagicMock()
    manager = ChunkedTranscriptionManager(capturer, "sk-test", on_chunk, interval=600)
    with patch("chunked_transcription.threading.Timer") as mock_timer_cls:
        mock_timer = MagicMock()
        mock_timer_cls.return_value = mock_timer
        manager.start()
        manager.stop()
    mock_timer.cancel.assert_called_once()
    assert manager._active is False


def test_stop_before_start_does_not_raise(capturer):
    """Calling stop() without start() must not raise."""
    manager = ChunkedTranscriptionManager(capturer, "sk-test", MagicMock(), interval=600)
    manager.stop()  # should not raise


# ---------------------------------------------------------------------------
# _process_chunk behaviour
# ---------------------------------------------------------------------------

def test_process_chunk_calls_take_chunk(capturer):
    """_process_chunk grabs frames from the capturer."""
    on_chunk = MagicMock()
    manager = ChunkedTranscriptionManager(capturer, "sk-test", on_chunk, interval=600)
    manager._active = True
    with patch("chunked_transcription.threading.Timer"), \
         patch("chunked_transcription.transcribe", return_value="text"):
        manager._process_chunk()
        time.sleep(0.15)
    capturer.take_chunk.assert_called_once()


def test_on_chunk_called_with_transcription_text(capturer):
    """on_chunk receives the text returned by transcribe()."""
    on_chunk = MagicMock()
    manager = ChunkedTranscriptionManager(capturer, "sk-test", on_chunk, interval=600)
    manager._active = True
    with patch("chunked_transcription.threading.Timer"), \
         patch("chunked_transcription.transcribe", return_value="Hej världen"):
        manager._process_chunk()
        time.sleep(0.3)
    on_chunk.assert_called_once_with("Hej världen")


def test_empty_frames_skip_transcription(capturer):
    """If take_chunk returns [], transcribe() is never called."""
    capturer.take_chunk.return_value = []
    on_chunk = MagicMock()
    manager = ChunkedTranscriptionManager(capturer, "sk-test", on_chunk, interval=600)
    manager._active = True
    with patch("chunked_transcription.threading.Timer"), \
         patch("chunked_transcription.transcribe") as mock_transcribe:
        manager._process_chunk()
        time.sleep(0.1)
    mock_transcribe.assert_not_called()
    on_chunk.assert_not_called()


def test_inactive_process_chunk_does_nothing(capturer):
    """_process_chunk exits immediately when _active is False."""
    on_chunk = MagicMock()
    manager = ChunkedTranscriptionManager(capturer, "sk-test", on_chunk, interval=600)
    manager._active = False
    with patch("chunked_transcription.transcribe") as mock_transcribe:
        manager._process_chunk()
        time.sleep(0.1)
    capturer.take_chunk.assert_not_called()
    mock_transcribe.assert_not_called()


def test_next_chunk_scheduled_after_processing(capturer):
    """After a chunk fires, a new Timer is created for the following chunk."""
    on_chunk = MagicMock()
    manager = ChunkedTranscriptionManager(capturer, "sk-test", on_chunk, interval=600)
    manager._active = True
    timer_instances = []
    with patch("chunked_transcription.threading.Timer") as mock_timer_cls:
        def make_timer(interval, fn):
            t = MagicMock()
            timer_instances.append(t)
            return t
        mock_timer_cls.side_effect = make_timer
        with patch("chunked_transcription.transcribe", return_value="x"):
            manager._process_chunk()
            time.sleep(0.2)
    # At least one new timer should have been created for the next chunk
    assert len(timer_instances) >= 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_transcription_error_calls_on_error(capturer):
    """If transcribe() raises, on_error is invoked with the message."""
    on_chunk = MagicMock()
    on_error = MagicMock()
    manager = ChunkedTranscriptionManager(
        capturer, "sk-test", on_chunk, on_error=on_error, interval=600
    )
    manager._active = True
    with patch("chunked_transcription.threading.Timer"), \
         patch("chunked_transcription.transcribe", side_effect=ValueError("API error")):
        manager._process_chunk()
        time.sleep(0.3)
    on_chunk.assert_not_called()
    on_error.assert_called_once()
    assert "API error" in on_error.call_args[0][0]


def test_transcription_error_without_on_error_does_not_raise(capturer):
    """If on_error is None, errors are silently swallowed."""
    manager = ChunkedTranscriptionManager(
        capturer, "sk-test", MagicMock(), on_error=None, interval=600
    )
    manager._active = True
    with patch("chunked_transcription.threading.Timer"), \
         patch("chunked_transcription.transcribe", side_effect=RuntimeError("boom")):
        manager._process_chunk()
        time.sleep(0.2)  # should not raise


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------

def test_flush_transcribes_remaining_frames(capturer):
    """flush() sends leftover frames to the API without waiting for the timer."""
    on_chunk = MagicMock()
    manager = ChunkedTranscriptionManager(capturer, "sk-test", on_chunk, interval=600)
    with patch("chunked_transcription.transcribe", return_value="Final chunk text"):
        manager.flush()
        time.sleep(0.3)
    on_chunk.assert_called_once_with("Final chunk text")


def test_flush_skips_empty_frames(capturer):
    """flush() with no frames in the buffer makes no API call."""
    capturer.take_chunk.return_value = []
    on_chunk = MagicMock()
    manager = ChunkedTranscriptionManager(capturer, "sk-test", on_chunk, interval=600)
    with patch("chunked_transcription.transcribe") as mock_transcribe:
        manager.flush()
        time.sleep(0.1)
    mock_transcribe.assert_not_called()
    on_chunk.assert_not_called()


def test_flush_calls_take_chunk(capturer):
    """flush() delegates to take_chunk to atomically drain the buffer."""
    on_chunk = MagicMock()
    manager = ChunkedTranscriptionManager(capturer, "sk-test", on_chunk, interval=600)
    with patch("chunked_transcription.transcribe", return_value="x"):
        manager.flush()
        time.sleep(0.2)
    capturer.take_chunk.assert_called_once()


# ---------------------------------------------------------------------------
# Integration: fast interval fires naturally
# ---------------------------------------------------------------------------

def test_chunk_fires_after_interval(capturer):
    """With a short interval, at least one chunk is transcribed automatically."""
    results = []
    manager = ChunkedTranscriptionManager(
        capturer,
        "sk-test",
        on_chunk=lambda t: results.append(t),
        interval=1,  # 1 second for testing
        language="sv",
    )
    with patch("chunked_transcription.transcribe", return_value="auto chunk"):
        manager.start()
        time.sleep(1.5)
        manager.stop()
    assert len(results) >= 1
    assert results[0] == "auto chunk"
