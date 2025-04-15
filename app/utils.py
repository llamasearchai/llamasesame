"""
Utility functions for LlamaSesame.

This module provides various utility functions used throughout the application.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger("llamasesame.utils")

# Default output directory
DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "VoiceCloneOutput")


def ensure_output_dir(output_dir: Optional[str] = None) -> str:
    """
    Ensure the output directory exists.

    Args:
        output_dir: Optional output directory path

    Returns:
        Path to the output directory
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir


def get_timestamp() -> str:
    """
    Get a formatted timestamp string.

    Returns:
        Formatted timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Loaded JSON data or None if an error occurred
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return None


def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Path to the output file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes} min {remaining_seconds:.2f} sec"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours} hr {minutes} min {remaining_seconds:.2f} sec"


def load_history(history_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load generation history from a file.

    Args:
        history_file: Optional path to the history file

    Returns:
        List of generation history entries
    """
    if history_file is None:
        history_file = os.path.join(DEFAULT_OUTPUT_DIR, "history.json")

    if os.path.exists(history_file):
        data = load_json(history_file)
        if data and isinstance(data, list):
            return data

    return []


def save_history(
    history: List[Dict[str, Any]], history_file: Optional[str] = None
) -> bool:
    """
    Save generation history to a file.

    Args:
        history: List of generation history entries
        history_file: Optional path to the history file

    Returns:
        True if successful, False otherwise
    """
    if history_file is None:
        ensure_output_dir(DEFAULT_OUTPUT_DIR)
        history_file = os.path.join(DEFAULT_OUTPUT_DIR, "history.json")

    return save_json(history, history_file)


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def is_audio_file(file_path: str) -> bool:
    """
    Check if a file is an audio file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        True if the file has an audio extension, False otherwise
    """
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    ext = os.path.splitext(file_path)[1].lower()
    return ext in audio_extensions
