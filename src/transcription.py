from pathlib import Path

import openai

from hallucination_filter import clean_transcript

# Whisper API file size limit (25 MB)
MAX_FILE_BYTES = 25 * 1024 * 1024

# Supported audio/video extensions accepted by Whisper API
SUPPORTED_EXTENSIONS = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm",
    ".ogg", ".flac", ".mov", ".avi", ".mkv",
}


def transcribe(file_path: str, api_key: str, language: str = "sv") -> str:
    """
    Transcribe *file_path* using OpenAI Whisper API.

    Parameters
    ----------
    file_path : str
        Path to a local audio or video file.
    api_key : str
        OpenAI API key.
    language : str
        BCP-47 language code. Defaults to "sv" (Swedish).

    Returns
    -------
    str
        Transcribed text.

    Raises
    ------
    ValueError
        If the file is missing, unsupported, or exceeds the size limit.
    openai.OpenAIError
        On API errors.
    """
    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    file_size = path.stat().st_size
    if file_size > MAX_FILE_BYTES:
        raise ValueError(
            f"File is {file_size / 1024 / 1024:.1f} MB, "
            f"which exceeds the 25 MB Whisper API limit."
        )

    client = openai.OpenAI(api_key=api_key)

    with open(file_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
        )

    return clean_transcript(response.text)
