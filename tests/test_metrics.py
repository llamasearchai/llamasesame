"""
Tests for the voice metrics functionality of LlamaSesame.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


def test_calculate_voice_metrics_integration():
    """Test the voice metrics calculation with mock data."""
    from llamasesame.app.metrics import calculate_voice_metrics

    # Mock the audio processing functions
    with mock.patch("llamasesame.app.metrics.librosa.load") as mock_load:
        # Create mock audio data
        mock_audio1 = np.random.rand(16000)  # 1 second of random audio
        mock_audio2 = np.random.rand(16000)  # 1 second of random audio

        # Configure the mock
        mock_load.side_effect = lambda file_path, sr=None: (
            mock_audio1 if "original" in file_path else mock_audio2,
            16000,
        )

        # Mock the similarity calculation functions
        with mock.patch(
            "llamasesame.app.metrics.calculate_pitch_similarity"
        ) as mock_pitch, mock.patch(
            "llamasesame.app.metrics.calculate_spectral_similarity"
        ) as mock_spectral, mock.patch(
            "llamasesame.app.metrics.calculate_mfcc_similarity"
        ) as mock_mfcc, mock.patch(
            "llamasesame.app.metrics.calculate_formant_similarity"
        ) as mock_formant:

            # Set the return values
            mock_pitch.return_value = 0.8
            mock_spectral.return_value = 0.75
            mock_mfcc.return_value = 0.9
            mock_formant.return_value = 0.85

            # Call the function with temporary file paths
            with tempfile.NamedTemporaryFile(
                suffix=".wav"
            ) as temp_original, tempfile.NamedTemporaryFile(
                suffix=".wav"
            ) as temp_generated:

                # Make the original file path contain "original" for our mock
                temp_original_path = temp_original.name.replace(".wav", "_original.wav")
                os.symlink(temp_original.name, temp_original_path)

                try:
                    # Calculate metrics
                    metrics = calculate_voice_metrics(
                        temp_original_path, temp_generated.name
                    )

                    # Check that the metrics were calculated
                    assert metrics is not None
                    assert isinstance(metrics, dict)

                    # Check that all expected metrics are present
                    assert "overall_similarity" in metrics
                    assert "pitch_similarity" in metrics
                    assert "spectral_similarity" in metrics
                    assert "mfcc_similarity" in metrics

                    # Check the values
                    assert metrics["pitch_similarity"] == 0.8
                    assert metrics["spectral_similarity"] == 0.75
                    assert metrics["mfcc_similarity"] == 0.9

                    # Check that overall similarity is the average
                    expected_overall = (0.8 + 0.75 + 0.9 + 0.85) / 4
                    assert metrics["overall_similarity"] == pytest.approx(
                        expected_overall
                    )
                finally:
                    # Clean up the symlink
                    if os.path.exists(temp_original_path):
                        os.unlink(temp_original_path)


def test_pitch_similarity():
    """Test the pitch similarity calculation."""
    from llamasesame.app.metrics import calculate_pitch_similarity

    # Create test audio data
    y_orig = np.sin(2 * np.pi * 440 * np.arange(0, 1, 1 / 16000))  # 440 Hz sine wave
    y_gen = np.sin(
        2 * np.pi * 440 * np.arange(0, 1, 1 / 16000)
    )  # Identical 440 Hz sine wave

    # Calculate similarity for identical signals
    similarity = calculate_pitch_similarity(y_orig, y_gen, 16000)

    # Should be close to 1.0 for identical signals
    assert similarity > 0.9

    # Create test audio data with different pitch
    y_orig = np.sin(2 * np.pi * 440 * np.arange(0, 1, 1 / 16000))  # 440 Hz sine wave
    y_gen = np.sin(
        2 * np.pi * 880 * np.arange(0, 1, 1 / 16000)
    )  # 880 Hz sine wave (octave higher)

    # Calculate similarity for very different signals
    similarity = calculate_pitch_similarity(y_orig, y_gen, 16000)

    # Should be lower for different pitches
    assert similarity < 0.9


def test_spectral_similarity():
    """Test the spectral similarity calculation."""
    from llamasesame.app.metrics import calculate_spectral_similarity

    # Create test audio data
    y_orig = np.sin(2 * np.pi * 440 * np.arange(0, 1, 1 / 16000))  # 440 Hz sine wave
    y_gen = np.sin(
        2 * np.pi * 440 * np.arange(0, 1, 1 / 16000)
    )  # Identical 440 Hz sine wave

    # Calculate similarity for identical signals
    similarity = calculate_spectral_similarity(y_orig, y_gen, 16000)

    # Should be close to 1.0 for identical signals
    assert similarity > 0.9

    # Create test audio data with different spectral content
    y_orig = np.sin(2 * np.pi * 440 * np.arange(0, 1, 1 / 16000))  # 440 Hz sine wave
    y_gen = 0.5 * np.sin(2 * np.pi * 440 * np.arange(0, 1, 1 / 16000)) + 0.5 * np.sin(
        2 * np.pi * 880 * np.arange(0, 1, 1 / 16000)
    )  # Mix of 440 Hz and 880 Hz

    # Calculate similarity for different signals
    similarity = calculate_spectral_similarity(y_orig, y_gen, 16000)

    # Should be lower for different spectral content
    assert similarity < 1.0


def test_mfcc_similarity():
    """Test the MFCC similarity calculation."""
    from llamasesame.app.metrics import calculate_mfcc_similarity

    # Create test audio data
    y_orig = np.sin(2 * np.pi * 440 * np.arange(0, 1, 1 / 16000))  # 440 Hz sine wave
    y_gen = np.sin(
        2 * np.pi * 440 * np.arange(0, 1, 1 / 16000)
    )  # Identical 440 Hz sine wave

    # Calculate similarity for identical signals
    similarity = calculate_mfcc_similarity(y_orig, y_gen, 16000)

    # Should be close to 1.0 for identical signals
    assert similarity > 0.9

    # Create test audio data with different spectral content
    y_orig = np.sin(2 * np.pi * 440 * np.arange(0, 1, 1 / 16000))  # 440 Hz sine wave
    y_gen = np.sin(2 * np.pi * 880 * np.arange(0, 1, 1 / 16000))  # 880 Hz sine wave

    # Calculate similarity for different signals
    similarity = calculate_mfcc_similarity(y_orig, y_gen, 16000)

    # Should be lower for different spectral content
    assert similarity < 1.0
