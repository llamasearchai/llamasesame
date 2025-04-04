#!/usr/bin/env python3
"""
LlamaSesame Voice Cloning Studio Pro

A state-of-the-art voice cloning platform using advanced AI technologies.
"""

import os
import time
import threading
import uuid
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import mesop as web

from llamasesame.app.core import clone_voice, batch_process
from llamasesame.app.models import init_models, list_available_models
from llamasesame.app.metrics import calculate_voice_metrics
from llamasesame.app.utils import (
    ensure_output_dir, get_timestamp, load_history, save_history,
    format_duration, truncate_text, is_audio_file
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.expanduser("~"), "VoiceCloneOutput", "app.log"), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("llamasesame")

# Global variables
history = []
current_model_id = "sesame/csm-1b"
batch_queue = []
is_batch_processing = False
settings = {
    "theme": "light",
    "advanced_mode": False,
    "enable_metrics": True,
    "cache_models": True,
    "sample_rate": 16000,
    "max_history": 20
}

# Load settings
def load_settings():
    global settings
    settings_path = os.path.join(os.path.expanduser("~"), "VoiceCloneOutput", "settings.json")
    try:
        if os.path.exists(settings_path):
            with open(settings_path, "r") as f:
                import json
                loaded_settings = json.load(f)
                settings.update(loaded_settings)
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")

# Save settings
def save_settings():
    settings_path = os.path.join(os.path.expanduser("~"), "VoiceCloneOutput", "settings.json")
    try:
        os.makedirs(os.path.dirname(settings_path), exist_ok=True)
        with open(settings_path, "w") as f:
            import json
            json.dump(settings, f)
    except Exception as e:
        logger.error(f"Failed to save settings: {e}")

# Process batch queue
def process_batch_queue():
    global is_batch_processing, batch_queue
    
    if is_batch_processing or not batch_queue:
        return
    
    is_batch_processing = True
    
    try:
        logger.info(f"Starting batch processing with {len(batch_queue)} jobs")
        
        for i, job in enumerate(batch_queue):
            logger.info(f"Processing batch job {i+1}/{len(batch_queue)}")
            
            try:
                output_path = clone_voice(
                    audio_file=job["audio_file"],
                    context_text=job["context_text"],
                    text=job["text"],
                    quality=job["quality"],
                    model_id=job.get("model_id", current_model_id),
                    temperature=job.get("temperature", 0.8),
                    num_beams=job.get("num_beams"),
                    output_filename=job.get("output_filename")
                )
                
                job["status"] = "completed"
                job["output_path"] = output_path
                job["completion_time"] = get_timestamp()
                
                # Add to history
                add_to_history(
                    reference_audio=job["audio_file"],
                    context_text=job["context_text"],
                    synthesis_text=job["text"],
                    output_path=output_path,
                    model_id=job.get("model_id", current_model_id),
                    quality_settings={
                        "quality": job["quality"],
                        "temperature": job.get("temperature", 0.8),
                        "num_beams": job.get("num_beams")
                    }
                )
                
            except Exception as e:
                logger.error(f"Error processing batch job {i+1}: {e}")
                job["status"] = "failed"
                job["error"] = str(e)
        
        # Clear completed and failed jobs
        batch_queue = [job for job in batch_queue if job["status"] == "pending"]
        
    finally:
        is_batch_processing = False


# Apply theme
def apply_theme():
    if settings["theme"] == "dark":
        web.style("""
        body {
            background-color: #121212;
            color: #e0e0e0;
        }
        .stTextInput, .stTextArea, .stSelectbox, .stNumberInput {
            background-color: #1e1e1e;
            color: #e0e0e0;
            border-color: #333;
        }
        .stButton>button {
            background-color: #4a4a4a;
            color: #ffffff;
        }
        .stButton>button:hover {
            background-color: #5a5a5a;
        }
        .main {
            background-color: #121212;
        }
        """)
    else:
        web.style("""
        body {
            background-color: #ffffff;
            color: #000000;
        }
        .stButton>button {
            background-color: #f0f2f6;
            color: #000000;
        }
        .stButton>button:hover {
            background-color: #e0e2e6;
        }
        """)
    
    # Add custom CSS from file
    try:
        with open("custom.css", "r") as f:
            custom_css = f.read()
            web.style(custom_css)
    except:
        pass


@web.route("/")
def home():
    """Main application UI."""
    web.title("LlamaSesame Voice Cloning Studio Pro")
    
    # Apply theme
    apply_theme()
    
    # Create tabs
    tabs = web.tabs(["Voice Cloning", "History", "Batch Processing", "Settings"])
    
    with tabs[0]:
        clone_voice_ui()
    
    with tabs[1]:
        history_ui()
    
    with tabs[2]:
        batch_processing_ui()
    
    with tabs[3]:
        settings_ui()


def clone_voice_ui():
    """Voice cloning interface."""
    global current_model_id
    
    web.header("Voice Cloning", level=2)
    web.markdown("""
    Upload a reference audio file and provide the text to generate speech with the cloned voice.
    For best results, use a clear audio recording with minimal background noise.
    """)
    
    col1, col2 = web.columns(2)
    
    with col1:
        web.subheader("Reference Audio")
        
        # Audio file upload
        uploaded_file = web.file_uploader(
            "Upload reference audio (WAV, MP3, FLAC)",
            type=["wav", "mp3", "flac", "ogg"]
        )
        
        # Context text (transcription of reference audio)
        context_text = web.text_area(
            "Context Text (transcription of reference audio)",
            height=100
        )
    
    with col2:
        web.subheader("Voice Generation")
        
        # Text to synthesize
        synthesis_text = web.text_area(
            "Text to Synthesize with Cloned Voice",
            height=100
        )
        
        # Model selection
        available_models = list_available_models()
        model_options = [{"label": details["name"], "value": model_id} 
                        for model_id, details in available_models.items()]
        
        selected_model = web.select(
            "Select Model",
            options=model_options,
            default=current_model_id
        )
        current_model_id = selected_model
        
        # Quality slider
        quality = web.slider(
            "Quality",
            min_value=1,
            max_value=10,
            value=5,
            help="Higher quality takes longer to generate but produces better results"
        )
    
    # Advanced settings
    with web.expander("Advanced Voice Settings", expanded=settings["advanced_mode"]):
        col3, col4 = web.columns(2)
        
        with col3:
            temperature = web.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Higher values produce more creative but less stable results"
            )
        
        with col4:
            num_beams = web.number_input(
                "Beam Count",
                min_value=1,
                max_value=10,
                value=None,
                help="Higher values produce more stable results but take longer (leave empty for auto)"
            )
        
        seed = web.number_input(
            "Random Seed",
            min_value=0,
            max_value=1000000,
            value=None,
            help="Set a seed for reproducible generation (leave empty for random)"
        )
    
    # Generate button
    col5, col6 = web.columns([2, 1])
    
    with col5:
        generate_button = web.button("Generate Voice", key="generate_voice")
    
    with col6:
        add_to_batch = web.button("Add to Batch Queue", key="add_to_batch")
    
    # Display status and results
    status_container = web.container()
    result_container = web.container()
    
    # Handle generation
    if generate_button:
        with status_container:
            web.info("Generating voice... This may take a few moments.")
            
            if not uploaded_file:
                web.error("Please upload a reference audio file.")
                return
            
            if not context_text:
                web.warning("Context text is empty. For best results, provide the transcription of the reference audio.")
            
            if not synthesis_text:
                web.error("Please enter text to synthesize.")
                return
            
            # Save uploaded file
            temp_dir = ensure_output_dir(os.path.join(os.path.expanduser("~"), "VoiceCloneOutput", "temp"))
            temp_file = os.path.join(temp_dir, f"reference_{get_timestamp()}_{uuid.uuid4().hex[:8]}{os.path.splitext(uploaded_file.name)[1]}")
            
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.read())
            
            try:
                # Generate voice
                output_path = clone_voice(
                    audio_file=temp_file,
                    context_text=context_text,
                    text=synthesis_text,
                    quality=quality,
                    model_id=current_model_id,
                    temperature=temperature,
                    num_beams=num_beams if num_beams else None,
                    seed=seed if seed else None,
                )
                
                # Add to history
                add_to_history(
                    reference_audio=temp_file,
                    context_text=context_text,
                    synthesis_text=synthesis_text,
                    output_path=output_path,
                    model_id=current_model_id,
                    quality_settings={
                        "quality": quality,
                        "temperature": temperature,
                        "num_beams": num_beams if num_beams else None,
                        "seed": seed if seed else None
                    }
                )
                
                with result_container:
                    web.success("Voice generation completed!")
                    
                    web.subheader("Generated Audio")
                    web.audio(output_path)
                    
                    web.markdown(f"**Output file:** `{os.path.basename(output_path)}`")
                    
                    if settings["enable_metrics"]:
                        web.subheader("Voice Metrics")
                        metrics = calculate_voice_metrics(temp_file, output_path)
                        
                        metric_cols = web.columns(4)
                        with metric_cols[0]:
                            web.metric("Overall Similarity", f"{metrics['overall_similarity']:.2f}")
                        with metric_cols[1]:
                            web.metric("Pitch Similarity", f"{metrics['pitch_similarity']:.2f}")
                        with metric_cols[2]:
                            web.metric("Spectral Similarity", f"{metrics['spectral_similarity']:.2f}")
                        with metric_cols[3]:
                            web.metric("MFCC Similarity", f"{metrics['mfcc_similarity']:.2f}")
            
            except Exception as e:
                with result_container:
                    web.error(f"Error generating voice: {e}")
    
    # Add to batch queue
    if add_to_batch:
        if not uploaded_file:
            web.error("Please upload a reference audio file.")
            return
        
        if not context_text:
            web.warning("Context text is empty. For best results, provide the transcription of the reference audio.")
        
        if not synthesis_text:
            web.error("Please enter text to synthesize.")
            return
        
        # Save uploaded file
        temp_dir = ensure_output_dir(os.path.join(os.path.expanduser("~"), "VoiceCloneOutput", "temp"))
        temp_file = os.path.join(temp_dir, f"reference_{get_timestamp()}_{uuid.uuid4().hex[:8]}{os.path.splitext(uploaded_file.name)[1]}")
        
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())
        
        # Add to batch queue
        batch_queue.append({
            "audio_file": temp_file,
            "context_text": context_text,
            "text": synthesis_text,
            "quality": quality,
            "model_id": current_model_id,
            "temperature": temperature,
            "num_beams": num_beams if num_beams else None,
            "seed": seed if seed else None,
            "status": "pending",
            "queued_time": get_timestamp()
        })
        
        web.success(f"Added to batch queue at position {len(batch_queue)}")


def history_ui():
    """History interface."""
    global history
    
    web.header("Generation History", level=2)
    web.markdown("View and manage your previous voice generation results.")
    
    # Load history if not loaded
    if not history:
        history = load_history()
    
    # Actions
    col1, col2 = web.columns([3, 1])
    
    with col1:
        search_term = web.text_input("Search history", placeholder="Filter by text...")
    
    with col2:
        export_button = web.button("Export History", key="export_history")
        clear_button = web.button("Clear History", key="clear_history")
    
    if export_button:
        export_path = os.path.join(os.path.expanduser("~"), "VoiceCloneOutput", f"history_export_{get_timestamp()}.json")
        if save_history(history, export_path):
            web.success(f"History exported to: {export_path}")
        else:
            web.error("Failed to export history.")
    
    if clear_button:
        if web.confirm("Are you sure you want to clear your entire history?", confirm_button="Yes, clear it"):
            history = []
            save_history(history)
            web.success("History cleared.")
    
    # Filter history
    filtered_history = history
    if search_term:
        filtered_history = [
            entry for entry in history
            if search_term.lower() in entry.get("context_text", "").lower() or
               search_term.lower() in entry.get("synthesis_text", "").lower()
        ]
    
    # Display history
    if not filtered_history:
        web.info("No generation history found.")
    else:
        for i, entry in enumerate(reversed(filtered_history)):
            with web.expander(
                f"{entry.get('timestamp', 'Unknown date')} - {truncate_text(entry.get('synthesis_text', 'No text'), 50)}",
                expanded=(i == 0)  # Expand only the most recent entry
            ):
                col3, col4 = web.columns(2)
                
                with col3:
                    web.subheader("Reference Audio")
                    if os.path.exists(entry.get("reference_audio", "")):
                        web.audio(entry["reference_audio"])
                    else:
                        web.warning("Reference audio file not available")
                    
                    web.markdown(f"**Context Text:**\n{entry.get('context_text', 'Not available')}")
                
                with col4:
                    web.subheader("Generated Audio")
                    if os.path.exists(entry.get("output_path", "")):
                        web.audio(entry["output_path"])
                    else:
                        web.warning("Generated audio file not available")
                    
                    web.markdown(f"**Synthesis Text:**\n{entry.get('synthesis_text', 'Not available')}")
                
                # Show metrics if available
                if entry.get("metrics"):
                    web.subheader("Voice Metrics")
                    metric_cols = web.columns(4)
                    with metric_cols[0]:
                        web.metric("Overall Similarity", f"{entry['metrics'].get('overall_similarity', 0):.2f}")
                    with metric_cols[1]:
                        web.metric("Pitch Similarity", f"{entry['metrics'].get('pitch_similarity', 0):.2f}")
                    with metric_cols[2]:
                        web.metric("Spectral Similarity", f"{entry['metrics'].get('spectral_similarity', 0):.2f}")
                    with metric_cols[3]:
                        web.metric("MFCC Similarity", f"{entry['metrics'].get('mfcc_similarity', 0):.2f}")
                
                # Show generation details
                web.markdown(f"**Model:** {entry.get('model_id', 'Unknown')}")
                web.markdown(f"**Quality Settings:** Quality={entry.get('quality_settings', {}).get('quality', 'Unknown')}, "
                          f"Temperature={entry.get('quality_settings', {}).get('temperature', 'Unknown')}, "
                          f"Beams={entry.get('quality_settings', {}).get('num_beams', 'Auto')}")
                web.markdown(f"**Duration:** {format_duration(entry.get('duration', 0))}")


def batch_processing_ui():
    """Batch processing interface."""
    global batch_queue, is_batch_processing
    
    web.header("Batch Processing", level=2)
    web.markdown("Queue multiple voice cloning jobs and process them sequentially.")
    
    # Status and controls
    status = "Running" if is_batch_processing else "Idle"
    status_color = "#4CAF50" if is_batch_processing else "#9E9E9E"
    
    col1, col2 = web.columns([3, 1])
    
    with col1:
        web.markdown(f"**Status:** <span style='color:{status_color};font-weight:bold'>{status}</span> - {len(batch_queue)} job(s) in queue", unsafe_allow_html=True)
    
    with col2:
        start_button = web.button("Start Processing", key="start_batch", disabled=is_batch_processing or not batch_queue)
        clear_button = web.button("Clear Queue", key="clear_batch", disabled=is_batch_processing)
    
    if start_button and not is_batch_processing and batch_queue:
        # Start batch processing in a separate thread
        thread = threading.Thread(target=process_batch_queue)
        thread.daemon = True
        thread.start()
        web.rerun()
    
    if clear_button and not is_batch_processing:
        batch_queue = []
        web.success("Batch queue cleared.")
    
    # Queue table
    if not batch_queue:
        web.info("Batch queue is empty. Add jobs from the Voice Cloning tab.")
    else:
        web.subheader("Queue")
        web.markdown("The following jobs are in the queue:")
        
        for i, job in enumerate(batch_queue):
            with web.expander(f"Job {i+1}: {truncate_text(job.get('text', 'No text'), 50)}", expanded=(i == 0)):
                col3, col4 = web.columns(2)
                
                with col3:
                    web.markdown(f"**Reference Audio:** {os.path.basename(job.get('audio_file', 'Unknown'))}")
                    web.markdown(f"**Context Text:** {truncate_text(job.get('context_text', 'None'), 200)}")
                
                with col4:
                    web.markdown(f"**Synthesis Text:** {truncate_text(job.get('text', 'None'), 200)}")
                    web.markdown(f"**Model:** {job.get('model_id', current_model_id)}")
                    web.markdown(f"**Quality:** {job.get('quality', 5)}")
                    web.markdown(f"**Temperature:** {job.get('temperature', 0.8)}")
                    web.markdown(f"**Beam Count:** {job.get('num_beams', 'Auto')}")
                
                web.markdown(f"**Status:** {job.get('status', 'pending').capitalize()}")
                web.markdown(f"**Queued at:** {job.get('queued_time', 'Unknown')}")
                
                if job.get('status') == 'completed':
                    web.markdown(f"**Completed at:** {job.get('completion_time', 'Unknown')}")
                    if os.path.exists(job.get('output_path', '')):
                        web.audio(job['output_path'])
                elif job.get('status') == 'failed':
                    web.error(f"Error: {job.get('error', 'Unknown error')}")


def settings_ui():
    """Settings interface."""
    global settings
    
    web.header("Settings", level=2)
    web.markdown("Configure the application settings.")
    
    # Theme selection
    col1, col2 = web.columns(2)
    
    with col1:
        theme = web.radio(
            "Theme",
            options=[
                {"label": "Light Mode", "value": "light"},
                {"label": "Dark Mode", "value": "dark"}
            ],
            default=settings["theme"]
        )
        
        if theme != settings["theme"]:
            settings["theme"] = theme
            save_settings()
            web.rerun()
    
    with col2:
        advanced_mode = web.checkbox(
            "Advanced Mode",
            value=settings["advanced_mode"],
            help="Show advanced options by default"
        )
        
        if advanced_mode != settings["advanced_mode"]:
            settings["advanced_mode"] = advanced_mode
            save_settings()
    
    # Audio and processing settings
    web.subheader("Audio Settings")
    
    col3, col4 = web.columns(2)
    
    with col3:
        sample_rate = web.select(
            "Sample Rate",
            options=[
                {"label": "16 kHz", "value": 16000},
                {"label": "22.05 kHz", "value": 22050},
                {"label": "44.1 kHz", "value": 44100},
                {"label": "48 kHz", "value": 48000}
            ],
            default=settings["sample_rate"]
        )
        
        if sample_rate != settings["sample_rate"]:
            settings["sample_rate"] = sample_rate
            save_settings()
    
    with col4:
        enable_metrics = web.checkbox(
            "Enable Voice Metrics",
            value=settings["enable_metrics"],
            help="Calculate similarity metrics between reference and generated audio"
        )
        
        if enable_metrics != settings["enable_metrics"]:
            settings["enable_metrics"] = enable_metrics
            save_settings()
    
    # Cache settings
    web.subheader("Cache Settings")
    
    cache_models = web.checkbox(
        "Cache Models",
        value=settings["cache_models"],
        help="Cache models to disk for faster loading times"
    )
    
    if cache_models != settings["cache_models"]:
        settings["cache_models"] = cache_models
        save_settings()
    
    max_history = web.slider(
        "Maximum History Entries",
        min_value=5,
        max_value=100,
        value=settings["max_history"],
        step=5,
        help="Maximum number of entries to keep in history"
    )
    
    if max_history != settings["max_history"]:
        settings["max_history"] = max_history
        save_settings()
    
    # Reset settings
    web.subheader("Reset")
    
    reset_button = web.button("Reset to Default Settings", key="reset_settings")
    
    if reset_button:
        if web.confirm("Are you sure you want to reset all settings to default values?", confirm_button="Yes, reset settings"):
            settings = {
                "theme": "light",
                "advanced_mode": False,
                "enable_metrics": True,
                "cache_models": True,
                "sample_rate": 16000,
                "max_history": 20
            }
            save_settings()
            web.success("Settings reset to defaults.")
            web.rerun()
    
    # About section
    with web.expander("About"):
        web.markdown("""
        ## LlamaSesame Voice Cloning Studio Pro
        
        Version: 1.0.0
        
        A state-of-the-art voice cloning platform using advanced AI technologies.
        Built with PyTorch, Transformers, and Mesop.
        
        © 2024 LlamaSearch
        """)


def add_to_history(
    reference_audio: str,
    context_text: str,
    synthesis_text: str,
    output_path: str,
    model_id: Optional[str] = None,
    quality_settings: Optional[Dict[str, Any]] = None
):
    """Add a generation entry to history."""
    global history
    
    # Calculate metrics if enabled
    metrics = None
    if settings["enable_metrics"] and os.path.exists(reference_audio) and os.path.exists(output_path):
        try:
            metrics = calculate_voice_metrics(reference_audio, output_path)
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
    
    # Create history entry
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": get_timestamp(),
        "reference_audio": reference_audio,
        "context_text": context_text,
        "synthesis_text": synthesis_text,
        "output_path": output_path,
        "model_id": model_id or current_model_id,
        "quality_settings": quality_settings or {},
        "metrics": metrics,
        "duration": 0  # Could add actual processing time if tracked
    }
    
    # Add to history
    history.append(entry)
    
    # Trim history if too long
    if len(history) > settings["max_history"]:
        history = history[-settings["max_history"]:]
    
    # Save history
    save_history(history)
    
    return entry


def initialize():
    """Initialize the application."""
    global history
    
    # Create output directory
    output_dir = ensure_output_dir()
    logger.info(f"Output directory: {output_dir}")
    
    # Load settings
    load_settings()
    logger.info(f"Settings loaded: {settings}")
    
    # Load history
    history = load_history()
    logger.info(f"Loaded {len(history)} history entries")
    
    # Initialize models
    try:
        model, processor = init_models()
        logger.info(f"Models initialized: {current_model_id}")
    except Exception as e:
        logger.error(f"Error initializing models: {e}")


if __name__ == "__main__":
    # Initialize the application
    initialize()
    
    # Start the web app
    web.start(home) 