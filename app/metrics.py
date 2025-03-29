"""
Voice metrics for LlamaSesame.

This module provides functions for calculating similarity metrics between audio samples.
"""

import numpy as np
import logging
import librosa
from typing import Dict, Optional, Tuple, List
from scipy.spatial.distance import cosine
from scipy import signal

# Configure logging
logger = logging.getLogger("llamasesame.metrics")


def calculate_voice_metrics(
    original_audio: str,
    generated_audio: str,
    sample_rate: int = 16000
) -> Dict[str, float]:
    """
    Calculate similarity metrics between original and generated audio.
    
    Args:
        original_audio: Path to the original reference audio
        generated_audio: Path to the generated audio
        sample_rate: Sample rate to use for analysis
        
    Returns:
        Dictionary of metrics and their values
    """
    logger.info(f"Calculating voice metrics between {original_audio} and {generated_audio}")
    metrics = {}
    
    try:
        # Load audio files
        y_orig, sr_orig = librosa.load(original_audio, sr=sample_rate)
        y_gen, sr_gen = librosa.load(generated_audio, sr=sample_rate)
        
        # Normalize both signals
        y_orig = y_orig / np.max(np.abs(y_orig))
        y_gen = y_gen / np.max(np.abs(y_gen))
        
        # Calculate pitch similarity
        pitch_sim = calculate_pitch_similarity(y_orig, y_gen, sr_orig)
        metrics['pitch_similarity'] = pitch_sim
        
        # Calculate spectral similarity
        spec_sim = calculate_spectral_similarity(y_orig, y_gen, sr_orig)
        metrics['spectral_similarity'] = spec_sim
        
        # Calculate MFCCs similarity
        mfcc_sim = calculate_mfcc_similarity(y_orig, y_gen, sr_orig)
        metrics['mfcc_similarity'] = mfcc_sim
        
        # Calculate overall similarity (weighted average)
        overall = (pitch_sim * 0.3 + spec_sim * 0.4 + mfcc_sim * 0.3)
        metrics['overall_similarity'] = overall
        
    except Exception as e:
        logger.error(f"Error calculating voice metrics: {e}")
        # Return default metrics on error
        metrics = {
            'pitch_similarity': 0.0,
            'spectral_similarity': 0.0,
            'mfcc_similarity': 0.0,
            'overall_similarity': 0.0,
            'error': str(e)
        }
    
    return metrics


def calculate_pitch_similarity(y_orig: np.ndarray, y_gen: np.ndarray, sr: int) -> float:
    """
    Calculate pitch similarity between two audio samples.
    
    Args:
        y_orig: Original audio signal
        y_gen: Generated audio signal
        sr: Sample rate
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        # Extract pitch (F0) for both signals
        f0_orig, voiced_flag_orig, _ = librosa.pyin(y_orig, 
                                                 fmin=librosa.note_to_hz('C2'), 
                                                 fmax=librosa.note_to_hz('C7'),
                                                 sr=sr)
        f0_gen, voiced_flag_gen, _ = librosa.pyin(y_gen, 
                                               fmin=librosa.note_to_hz('C2'), 
                                               fmax=librosa.note_to_hz('C7'),
                                               sr=sr)
        
        # Filter out unvoiced frames
        f0_orig = f0_orig[voiced_flag_orig]
        f0_gen = f0_gen[voiced_flag_gen]
        
        # If either signal has no voiced frames, return 0
        if len(f0_orig) == 0 or len(f0_gen) == 0:
            return 0.0
        
        # Calculate the average absolute difference between pitch contours
        # First, interpolate to equal length
        min_len = min(len(f0_orig), len(f0_gen))
        f0_orig_interp = np.interp(np.linspace(0, 1, min_len), 
                                np.linspace(0, 1, len(f0_orig)), 
                                f0_orig)
        f0_gen_interp = np.interp(np.linspace(0, 1, min_len), 
                               np.linspace(0, 1, len(f0_gen)), 
                               f0_gen)
        
        # Calculate normalized distance
        diff = np.abs(f0_orig_interp - f0_gen_interp)
        pitch_dist = np.mean(diff) / np.mean(f0_orig_interp)
        
        # Convert to similarity score (1 - normalized distance)
        similarity = max(0, 1 - pitch_dist)
        return min(similarity, 1.0)  # Ensure result is between 0 and 1
        
    except Exception as e:
        logger.error(f"Error calculating pitch similarity: {e}")
        return 0.0


def calculate_spectral_similarity(y_orig: np.ndarray, y_gen: np.ndarray, sr: int) -> float:
    """
    Calculate spectral similarity between two audio samples.
    
    Args:
        y_orig: Original audio signal
        y_gen: Generated audio signal
        sr: Sample rate
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        # Compute spectrograms
        S_orig = np.abs(librosa.stft(y_orig))
        S_gen = np.abs(librosa.stft(y_gen))
        
        # Convert to log-mel spectrograms for better perceptual comparison
        mel_orig = librosa.feature.melspectrogram(S=S_orig, sr=sr)
        mel_gen = librosa.feature.melspectrogram(S=S_gen, sr=sr)
        
        # Convert to dB scale
        log_mel_orig = librosa.power_to_db(mel_orig, ref=np.max)
        log_mel_gen = librosa.power_to_db(mel_gen, ref=np.max)
        
        # Ensure equal lengths by padding/truncating
        min_width = min(log_mel_orig.shape[1], log_mel_gen.shape[1])
        log_mel_orig = log_mel_orig[:, :min_width]
        log_mel_gen = log_mel_gen[:, :min_width]
        
        # Flatten and normalize
        flat_orig = log_mel_orig.flatten()
        flat_gen = log_mel_gen.flatten()
        
        # Calculate cosine similarity
        similarity = 1 - cosine(flat_orig, flat_gen)
        
        return max(0, min(similarity, 1.0))  # Ensure result is between 0 and 1
        
    except Exception as e:
        logger.error(f"Error calculating spectral similarity: {e}")
        return 0.0


def calculate_mfcc_similarity(y_orig: np.ndarray, y_gen: np.ndarray, sr: int) -> float:
    """
    Calculate MFCC similarity between two audio samples.
    
    Args:
        y_orig: Original audio signal
        y_gen: Generated audio signal
        sr: Sample rate
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        # Extract MFCCs
        mfcc_orig = librosa.feature.mfcc(y=y_orig, sr=sr, n_mfcc=13)
        mfcc_gen = librosa.feature.mfcc(y=y_gen, sr=sr, n_mfcc=13)
        
        # Calculate delta features
        delta_orig = librosa.feature.delta(mfcc_orig)
        delta_gen = librosa.feature.delta(mfcc_gen)
        
        # Combine static and delta features
        features_orig = np.vstack([mfcc_orig, delta_orig])
        features_gen = np.vstack([mfcc_gen, delta_gen])
        
        # Ensure equal lengths
        min_width = min(features_orig.shape[1], features_gen.shape[1])
        features_orig = features_orig[:, :min_width]
        features_gen = features_gen[:, :min_width]
        
        # Flatten features
        flat_orig = features_orig.flatten()
        flat_gen = features_gen.flatten()
        
        # Calculate cosine similarity
        similarity = 1 - cosine(flat_orig, flat_gen)
        
        return max(0, min(similarity, 1.0))  # Ensure result is between 0 and 1
        
    except Exception as e:
        logger.error(f"Error calculating MFCC similarity: {e}")
        return 0.0


def calculate_formant_similarity(y_orig: np.ndarray, y_gen: np.ndarray, sr: int) -> float:
    """
    Calculate formant similarity between two audio samples.
    
    Args:
        y_orig: Original audio signal
        y_gen: Generated audio signal
        sr: Sample rate
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        # Extract formants (simplified by using LPC)
        lpc_orig = librosa.lpc(y_orig, order=16)
        lpc_gen = librosa.lpc(y_gen, order=16)
        
        # Calculate frequencies from LPC coefficients
        freqs_orig, _ = signal.freqz(1.0, lpc_orig)
        freqs_gen, _ = signal.freqz(1.0, lpc_gen)
        
        # Convert to Hz
        hz_freqs_orig = freqs_orig * sr / (2 * np.pi)
        hz_freqs_gen = freqs_gen * sr / (2 * np.pi)
        
        # Keep only valid frequencies (below Nyquist)
        valid_idx = hz_freqs_orig < (sr / 2)
        hz_freqs_orig = hz_freqs_orig[valid_idx]
        hz_freqs_gen = hz_freqs_gen[valid_idx]
        
        # Calculate similarity between formant distributions
        similarity = 1 - cosine(hz_freqs_orig, hz_freqs_gen)
        
        return max(0, min(similarity, 1.0))  # Ensure result is between 0 and 1
        
    except Exception as e:
        logger.error(f"Error calculating formant similarity: {e}")
        return 0.0 