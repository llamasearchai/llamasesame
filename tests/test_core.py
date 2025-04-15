"""
Tests for the core functionality of LlamaSesame.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# Mock the core module
@mock.patch("llamasesame.app.core.init_models")
def test_clone_voice_basic_functionality(mock_init_models):
    """Test that clone_voice works with basic parameters."""
    from llamasesame.app.core import clone_voice

    # Create temp files for testing
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
        # Mock the model and processor
        mock_model = mock.MagicMock()
        mock_processor = mock.MagicMock()
        mock_init_models.return_value = (mock_model, mock_processor)

        # Mock the audio processing functions
        with mock.patch("llamasesame.app.core.torchaudio.load") as mock_load:
            mock_load.return_value = (mock.MagicMock(), 16000)

            # Mock the model output
            mock_model.return_value = mock.MagicMock()

            # Mock the save function
            with mock.patch("llamasesame.app.core.torchaudio.save") as mock_save:
                mock_save.return_value = None

                # Call the function
                output_path = clone_voice(
                    audio_file=temp_audio.name,
                    context_text="Test context",
                    text="This is a test",
                    quality=5,
                )

                # Check that the function called the model
                assert mock_init_models.called
                assert mock_load.called
                assert mock_save.called

                # Check that the output path is valid
                assert isinstance(output_path, str)
                assert output_path.endswith(".wav")
                assert "VoiceCloneOutput" in output_path


@mock.patch("llamasesame.app.core.init_models")
def test_batch_process(mock_init_models):
    """Test that batch processing works."""
    from llamasesame.app.core import batch_process

    # Create temp files for testing
    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as temp_audio1, tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio2:

        # Mock the model and processor
        mock_model = mock.MagicMock()
        mock_processor = mock.MagicMock()
        mock_init_models.return_value = (mock_model, mock_processor)

        # Mock the clone_voice function
        with mock.patch("llamasesame.app.core.clone_voice") as mock_clone_voice:
            mock_clone_voice.side_effect = (
                lambda **kwargs: f"/tmp/output_{kwargs['text']}.wav"
            )

            # Create test jobs
            jobs = [
                {
                    "audio_file": temp_audio1.name,
                    "context_text": "Context 1",
                    "text": "Text 1",
                    "quality": 5,
                },
                {
                    "audio_file": temp_audio2.name,
                    "context_text": "Context 2",
                    "text": "Text 2",
                    "quality": 7,
                },
            ]

            # Process the batch
            results = batch_process(jobs)

            # Check that the function called clone_voice twice
            assert mock_clone_voice.call_count == 2

            # Check the results
            assert len(results) == 2
            assert results[0]["status"] == "completed"
            assert results[0]["output_path"] == "/tmp/output_Text 1.wav"
            assert results[1]["status"] == "completed"
            assert results[1]["output_path"] == "/tmp/output_Text 2.wav"


def test_input_validation():
    """Test input validation in core functions."""
    from llamasesame.app.core import clone_voice

    # Test with invalid audio file
    with pytest.raises(FileNotFoundError):
        clone_voice(
            audio_file="/nonexistent/file.wav",
            context_text="Test context",
            text="This is a test",
            quality=5,
        )

    # Test with invalid quality
    with pytest.raises(ValueError):
        clone_voice(
            audio_file="tests/data/test.wav",
            context_text="Test context",
            text="This is a test",
            quality=11,  # Invalid quality (must be 1-10)
        )

    # Test with empty text
    with pytest.raises(ValueError):
        clone_voice(
            audio_file="tests/data/test.wav",
            context_text="Test context",
            text="",  # Empty text
            quality=5,
        )
