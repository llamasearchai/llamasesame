#!/usr/bin/env python3
"""
LlamaSesame Command Line Interface

Provides command-line access to voice cloning functionality.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from llamasesame.app.core import batch_process, clone_voice
from llamasesame.app.models import init_models, list_available_models
from llamasesame.app.utils import ensure_output_dir, get_timestamp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("llamasesame.cli")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LlamaSesame Voice Cloning - Command Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Clone voice command
    clone_parser = subparsers.add_parser(
        "clone", help="Clone a voice from a reference audio"
    )
    clone_parser.add_argument(
        "--audio", "-a", required=True, help="Path to reference audio file"
    )
    clone_parser.add_argument(
        "--context",
        "-c",
        required=False,
        help="Context text (transcription of reference audio)",
    )
    clone_parser.add_argument(
        "--text", "-t", required=True, help="Text to synthesize with cloned voice"
    )
    clone_parser.add_argument(
        "--model", "-m", default="sesame/csm-1b", help="Model ID to use"
    )
    clone_parser.add_argument(
        "--quality",
        "-q",
        type=int,
        default=5,
        choices=range(1, 11),
        help="Quality level (1-10)",
    )
    clone_parser.add_argument(
        "--temperature", type=float, default=0.8, help="Generation temperature"
    )
    clone_parser.add_argument(
        "--beams", type=int, help="Number of beams for generation"
    )
    clone_parser.add_argument(
        "--seed", type=int, help="Random seed for reproducibility"
    )
    clone_parser.add_argument("--output", "-o", help="Output filename")
    clone_parser.add_argument("--output-dir", help="Output directory")

    # Batch processing command
    batch_parser = subparsers.add_parser(
        "batch", help="Process a batch of voice cloning jobs"
    )
    batch_parser.add_argument(
        "--file", "-f", required=True, help="JSON file with batch jobs"
    )
    batch_parser.add_argument(
        "--model", "-m", help="Default model ID to use (overrides job-specific model)"
    )
    batch_parser.add_argument("--output-dir", help="Output directory")

    # List models command
    list_parser = subparsers.add_parser(
        "list-models", help="List available voice cloning models"
    )

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")

    return parser.parse_args()


def display_models():
    """Display available voice cloning models."""
    models = list_available_models()
    print("\nAvailable Voice Cloning Models:")
    print("=" * 80)
    print(f"{'Model ID':<30} {'Name':<30} {'Description'}")
    print("-" * 80)

    for model_id, details in models.items():
        print(f"{model_id:<30} {details['name']:<30} {details['description']}")

    print("=" * 80)
    print("\nUse a Model ID with the --model argument to select a specific model.")


def display_version():
    """Display version information."""
    print("\nLlamaSesame Voice Cloning")
    print("Version: 1.0.0")
    print("© 2024 LlamaSearch")
    print("\nA state-of-the-art voice cloning platform using advanced AI technologies.")


def batch_from_file(
    file_path: str, model_id: Optional[str] = None, output_dir: Optional[str] = None
):
    """Process batch jobs from a JSON file."""
    import json

    try:
        with open(file_path, "r") as f:
            jobs = json.load(f)

        # Verify jobs format
        if not isinstance(jobs, list):
            raise ValueError("Batch file must contain a list of jobs")

        # Initialize models
        init_models(model_id)

        # Process batch
        results = batch_process(jobs, model_id, output_dir)

        # Display results
        print("\nBatch Processing Results:")
        print("=" * 80)
        for i, result in enumerate(results):
            status = "Completed" if result.get("status") == "completed" else "Failed"
            print(f"Job {i+1}: {status}")
            if status == "Completed":
                print(f"  Output: {result.get('output_path')}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")

        print("=" * 80)
        print(f"\nProcessed {len(results)} job(s)")

    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error processing batch file: {e}")
        sys.exit(1)


def main():
    """Main entry point for the command line interface."""
    args = parse_args()

    if args.command == "clone":
        # Check if audio file exists
        if not os.path.exists(args.audio):
            logger.error(f"Audio file not found: {args.audio}")
            sys.exit(1)

        try:
            # Initialize models
            init_models(args.model)

            # Clone voice
            output_path = clone_voice(
                audio_file=args.audio,
                context_text=args.context,
                text=args.text,
                quality=args.quality,
                model_id=args.model,
                temperature=args.temperature,
                num_beams=args.beams,
                seed=args.seed,
                output_filename=args.output,
                output_dir=args.output_dir,
            )

            logger.info(f"Voice cloning completed successfully!")
            logger.info(f"Output saved to: {output_path}")

        except Exception as e:
            logger.error(f"Error cloning voice: {e}")
            sys.exit(1)

    elif args.command == "batch":
        # Check if batch file exists
        if not os.path.exists(args.file):
            logger.error(f"Batch file not found: {args.file}")
            sys.exit(1)

        batch_from_file(args.file, args.model, args.output_dir)

    elif args.command == "list-models":
        display_models()

    elif args.command == "version":
        display_version()

    else:
        # If no command is provided, show help
        parse_args(["--help"])


if __name__ == "__main__":
    main()
