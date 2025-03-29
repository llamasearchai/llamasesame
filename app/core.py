"""
Core functionality for LlamaSesame voice cloning.

This module contains the main voice cloning functionality.
"""

import os
import time
import uuid
import logging
import torch
import torchaudio
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .models import init_models
from .metrics import calculate_voice_metrics
from .utils import ensure_output_dir, get_timestamp

# Configure logging
logger = logging.getLogger("llamasesame.core")

# Default output directory
DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "VoiceCloneOutput")


def clone_voice(
    audio_file: str,
    context_text: str,
    text: str,
    quality: int = 5,
    output_filename: Optional[str] = None,
    model_id: Optional[str] = None,
    temperature: float = 0.8,
    num_beams: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    compute_metrics: bool = True,
) -> str:
    """
    Clone a voice from a reference audio file and generate speech with the cloned voice.
    
    Args:
        audio_file: Path to the reference audio file
        context_text: Transcription of the reference audio
        text: Text to synthesize with the cloned voice
        quality: Quality level (1-10, higher is better but slower)
        output_filename: Optional custom filename for the output
        model_id: Optional model ID to use (default is the current global model)
        temperature: Temperature for generation (higher = more creative but less stable)
        num_beams: Number of beams for beam search (None = use quality-based default)
        seed: Random seed for reproducibility
        output_dir: Output directory for generated audio
        compute_metrics: Whether to compute voice metrics
        
    Returns:
        Path to the generated audio file
    """
    start_time = time.time()
    logger.info(f"Starting voice cloning process")
    
    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        logger.info(f"Set random seed to {seed}")
    
    # Initialize models
    model, processor = init_models(model_id)
    
    # Determine quality settings
    if num_beams is None:
        num_beams = max(1, quality * 2)  # Scale beams with quality
    
    # Prepare output directory and filename
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    if output_filename is None:
        timestamp = get_timestamp()
        output_filename = f"generated_{timestamp}_{uuid.uuid4().hex[:8]}.wav"
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Load the audio file
    logger.info(f"Loading reference audio: {audio_file}")
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Prepare inputs for the model
    logger.info("Preparing inputs for voice cloning")
    inputs = processor(
        audio=waveform.squeeze().numpy(),
        sampling_rate=sample_rate,
        text=context_text,
        padding=True,
        return_tensors="pt"
    )
    
    # Generate speech with the cloned voice
    logger.info(f"Generating speech with quality {quality}, temperature {temperature}, and {num_beams} beams")
    with torch.no_grad():
        generated_audio = model.generate(
            **inputs,
            prompt_text=text,
            temperature=temperature,
            num_beams=num_beams,
            do_sample=(temperature > 0),
            max_new_tokens=500 * quality  # Scale token length with quality
        )
    
    # Save the generated audio
    logger.info(f"Saving generated audio to {output_path}")
    torchaudio.save(
        output_path,
        torch.tensor(generated_audio.cpu().numpy()).unsqueeze(0),
        sample_rate
    )
    
    # Calculate metrics if requested
    metrics = None
    if compute_metrics:
        logger.info("Calculating voice similarity metrics")
        metrics = calculate_voice_metrics(
            original_audio=audio_file,
            generated_audio=output_path
        )
        logger.info(f"Voice metrics: {metrics}")
    
    # Log completion time
    duration = time.time() - start_time
    logger.info(f"Voice cloning completed in {duration:.2f} seconds")
    
    return output_path


def batch_process(
    jobs: list,
    model_id: Optional[str] = None,
    output_dir: Optional[str] = None
) -> list:
    """
    Process multiple voice cloning jobs in batch.
    
    Args:
        jobs: List of job dictionaries with parameters for clone_voice
        model_id: Optional model ID to use for all jobs
        output_dir: Optional output directory for all generated files
        
    Returns:
        List of output paths for the generated audio files
    """
    results = []
    
    logger.info(f"Starting batch processing of {len(jobs)} jobs")
    for i, job in enumerate(jobs):
        logger.info(f"Processing job {i+1}/{len(jobs)}")
        
        # Use provided model_id as default if not specified in job
        if model_id and "model_id" not in job:
            job["model_id"] = model_id
            
        # Use provided output_dir as default if not specified in job
        if output_dir and "output_dir" not in job:
            job["output_dir"] = output_dir
        
        try:
            output_path = clone_voice(**job)
            results.append({
                "status": "success",
                "output_path": output_path,
                "job": job
            })
        except Exception as e:
            logger.error(f"Error processing job {i+1}: {e}")
            results.append({
                "status": "error",
                "error": str(e),
                "job": job
            })
    
    logger.info(f"Batch processing completed. {len([r for r in results if r['status'] == 'success'])} successful, "
                f"{len([r for r in results if r['status'] == 'error'])} failed")
    
    return results 