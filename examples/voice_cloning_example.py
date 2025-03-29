#!/usr/bin/env python3
"""
LlamaSesame Voice Cloning Example

This script demonstrates how to use the LlamaSesame library programmatically.
"""

import os
import time
from pathlib import Path

from llamasesame import clone_voice, init_models, calculate_voice_metrics

def main():
    """Run a simple voice cloning example."""
    print("LlamaSesame Voice Cloning Example")
    print("=================================")
    
    # Set up paths
    examples_dir = Path(__file__).parent
    audio_dir = examples_dir / "audio"
    output_dir = examples_dir / "output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Path to sample audio file
    audio_file = str(audio_dir / "sample1.wav")
    
    # Check if sample audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Sample audio file not found at {audio_file}")
        print("Please add a WAV file at this location or update the path in this script.")
        return
    
    # Initialize models (this loads the model and processor)
    print("\nInitializing models...")
    init_models()
    
    # Sample text for cloning
    context_text = "This is a sample voice recording for demonstration."
    synthesis_text = "Hello, I'm speaking with a cloned voice. This technology is amazing!"
    
    # Clone the voice
    print("\nCloning voice...")
    start_time = time.time()
    
    output_path = clone_voice(
        audio_file=audio_file,
        context_text=context_text,
        text=synthesis_text,
        quality=7,  # Higher quality (1-10)
        temperature=0.7,  # Lower for more stable output
        output_dir=str(output_dir)
    )
    
    duration = time.time() - start_time
    print(f"Voice cloning completed in {duration:.2f} seconds!")
    print(f"Generated audio saved to: {output_path}")
    
    # Calculate voice metrics
    print("\nCalculating voice similarity metrics...")
    metrics = calculate_voice_metrics(audio_file, output_path)
    
    print("\nVoice Metrics:")
    print(f"- Overall Similarity: {metrics['overall_similarity']:.2f}")
    print(f"- Pitch Similarity: {metrics['pitch_similarity']:.2f}")
    print(f"- Spectral Similarity: {metrics['spectral_similarity']:.2f}")
    print(f"- MFCC Similarity: {metrics['mfcc_similarity']:.2f}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 