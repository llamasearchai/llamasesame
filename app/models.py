"""
Model management for LlamaSesame.

This module handles loading, caching, and management of voice cloning models.
"""

import os
import hashlib
import logging
import torch
from typing import Dict, Any, Optional, Tuple
from transformers import AutoProcessor, AutoModelForTextToWaveform, set_seed

# Configure logging
logger = logging.getLogger("llamasesame.models")

# Global variables
models = {}
processors = {}
current_model_id = "sesame/csm-1b"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".voice_cloning_cache")

# Available models
AVAILABLE_MODELS = {
    "sesame/csm-1b": {
        "name": "CloneMyVoice (CSM-1B)",
        "description": "High-quality voice cloning model",
        "requires_auth": True
    },
    "cvssp/sesame-ft": {
        "name": "SESAME Fine-tuned",
        "description": "Fine-tuned version with improved quality",
        "requires_auth": True
    }
}


def load_api_key() -> Optional[str]:
    """
    Load API key from environment or file.
    
    Returns:
        API key string or None if not found
    """
    # Try to get from environment first
    hf_token = os.environ.get("HF_TOKEN")
    
    # If not in environment, try to read from file
    if not hf_token:
        try:
            with open("apikeys.txt", "r") as file:
                for line in file:
                    if line.startswith("HF:"):
                        hf_token = line.strip().split(" ")[1]
        except FileNotFoundError:
            pass
    
    return hf_token


def get_cache_path(model_id: str) -> str:
    """
    Get cache path for a model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Path to the cache file
    """
    hash_id = hashlib.md5(model_id.encode()).hexdigest()
    return os.path.join(CACHE_DIR, hash_id)


def is_model_cached(model_id: str) -> bool:
    """
    Check if model is cached.
    
    Args:
        model_id: Model identifier
        
    Returns:
        True if model is cached, False otherwise
    """
    cache_path = get_cache_path(model_id)
    return os.path.exists(cache_path)


def cache_model(model_id: str, model, processor):
    """
    Cache model to disk.
    
    Args:
        model_id: Model identifier
        model: Model instance
        processor: Processor instance
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = get_cache_path(model_id)
    try:
        torch.save({
            'model': model.state_dict(),
            'processor': processor
        }, cache_path)
        logger.info(f"Model {model_id} cached successfully")
    except Exception as e:
        logger.error(f"Failed to cache model {model_id}: {e}")


def load_model_from_cache(model_id: str) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Load model from cache.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Tuple of (model, processor) or (None, None) if loading fails
    """
    cache_path = get_cache_path(model_id)
    try:
        if os.path.exists(cache_path):
            logger.info(f"Loading model {model_id} from cache")
            checkpoint = torch.load(cache_path)
            model = AutoModelForTextToWaveform.from_pretrained(model_id)
            model.load_state_dict(checkpoint['model'])
            processor = checkpoint['processor']
            return model, processor
    except Exception as e:
        logger.error(f"Failed to load model {model_id} from cache: {e}")
    
    return None, None


def init_models(model_id: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Initialize models for voice cloning.
    
    Args:
        model_id: Optional model ID to use
        
    Returns:
        Tuple of (model, processor)
    """
    global current_model_id, models, processors
    
    # Use default model if none specified
    if model_id is None:
        model_id = current_model_id
    else:
        current_model_id = model_id
    
    # Check if model is already loaded
    if model_id in models and model_id in processors:
        logger.info(f"Using already loaded model: {model_id}")
        return models[model_id], processors[model_id]
    
    # Try to load from cache
    model, processor = load_model_from_cache(model_id)
    
    # If not in cache, load from Hugging Face
    if model is None or processor is None:
        logger.info(f"Loading model {model_id} from Hugging Face")
        
        # Load API key for authenticated models
        hf_token = None
        if model_id in AVAILABLE_MODELS and AVAILABLE_MODELS[model_id].get("requires_auth", False):
            hf_token = load_api_key()
            if not hf_token:
                logger.warning(f"No API key found for authenticated model {model_id}")
        
        # Load the model and processor
        try:
            processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
            model = AutoModelForTextToWaveform.from_pretrained(model_id, token=hf_token)
            
            # Cache the model for future use
            cache_model(model_id, model, processor)
        except Exception as e:
            logger.error(f"Failed to load model {model_id} from Hugging Face: {e}")
            raise
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Store in global cache
    models[model_id] = model
    processors[model_id] = processor
    
    return model, processor


def list_available_models():
    """
    List all available voice cloning models.
    
    Returns:
        Dictionary of available models with their metadata
    """
    return AVAILABLE_MODELS 