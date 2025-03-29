"""
LlamaSesame Voice Cloning Studio Pro

A state-of-the-art voice cloning platform using advanced AI technologies.
"""

__version__ = "1.0.0"

# Import core functionality for easier access
from llamasesame.app.core import clone_voice, batch_process
from llamasesame.app.models import init_models, list_available_models
from llamasesame.app.metrics import calculate_voice_metrics

# Define public API
__all__ = [
    "clone_voice",
    "batch_process",
    "init_models",
    "list_available_models",
    "calculate_voice_metrics",
] 