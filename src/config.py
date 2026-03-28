import os
from pathlib import Path

from dotenv import load_dotenv, set_key

ENV_FILE = Path(__file__).parent.parent / ".env"


def load_api_key() -> str:
    """Load OpenAI API key from .env file."""
    load_dotenv(ENV_FILE, override=True)
    return os.getenv("OPENAI_API_KEY", "")


def save_api_key(api_key: str) -> None:
    """Save OpenAI API key to .env file."""
    if not ENV_FILE.exists():
        ENV_FILE.touch()
    set_key(str(ENV_FILE), "OPENAI_API_KEY", api_key)
    os.environ["OPENAI_API_KEY"] = api_key
