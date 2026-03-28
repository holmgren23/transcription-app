"""Entry point for the Swedish Lecture Transcription app."""

import os
import sys

# Ensure src/ is on the path when run from the project root
sys.path.insert(0, os.path.dirname(__file__))

import tkinter as tk
from ui import TranscriptionApp


def main() -> None:
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
