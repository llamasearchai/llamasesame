# LlamaSesame Voice Cloning Studio Pro

A state-of-the-art voice cloning platform using advanced AI technologies.

![LlamaSesame Logo](https://llamasearch.ai

## 🌟 Features

- **High-Quality Voice Cloning**: Generate human-like speech that matches the vocal characteristics of a reference audio
- **User-Friendly Interface**: Simple web interface makes voice cloning accessible to everyone
- **Batch Processing**: Queue multiple voice cloning jobs for efficient processing
- **Advanced Settings**: Fine-tune generation parameters for optimal results
- **Voice Metrics**: Analyze similarity between original and cloned voices
- **Multiple Models**: Support for various voice cloning models
- **Cross-Platform**: Works on Linux, macOS, and Windows

## 📋 Requirements

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- CUDA-compatible GPU recommended for faster processing

## 🚀 Installation

### Using pip

```bash
pip install llamasesame
```

### From source

```bash
git clone https://llamasearch.ai
cd llamasesame
pip install -e .
```

### Using Docker

```bash
docker pull llamasearch/llamasesame:latest
docker run -p 8501:8501 llamasearch/llamasesame:latest
```

Or with Docker Compose:

```bash
docker-compose up
```

## 🖥️ Usage

### Web Interface

Launch the web interface with:

```bash
llamasesame-app
```

Then open your browser and navigate to `http://localhost:8501`

### Command Line Interface

Clone a voice:

```bash
llamasesame clone --audio reference.wav --text "Hello, this is a cloned voice." --quality 8
```

Process a batch of jobs:

```bash
llamasesame batch --file jobs.json
```

List available models:

```bash
llamasesame list-models
```

### Python API

```python
from llamasesame import clone_voice, init_models

# Initialize models
init_models()

# Clone a voice
output_path = clone_voice(
    audio_file="reference.wav",
    context_text="This is the original audio content.",
    text="This is the text I want to synthesize with the cloned voice.",
    quality=8
)

print(f"Generated audio saved to: {output_path}")
```

## 📊 Voice Metrics

LlamaSesame provides several metrics to evaluate the quality of voice cloning:

- **Overall Similarity**: Combined score of all metrics
- **Pitch Similarity**: How closely the pitch contour matches
- **Spectral Similarity**: Similarity in frequency domain characteristics
- **MFCC Similarity**: Similarity in voice timbre and pronunciation

## 🔧 Configuration

LlamaSesame settings can be configured through the web interface or by editing the settings file at `~/VoiceCloneOutput/settings.json`.

## 🔍 Advanced Usage

### Batch Job Format

Batch jobs are defined in a JSON file with the following format:

```json
[
  {
    "audio_file": "/path/to/reference1.wav",
    "context_text": "This is the transcription of reference1.wav",
    "text": "This is the text to synthesize with the first voice",
    "quality": 7,
    "model_id": "sesame/csm-1b"
  },
  {
    "audio_file": "/path/to/reference2.mp3",
    "context_text": "This is the transcription of reference2.mp3",
    "text": "This is the text to synthesize with the second voice",
    "quality": 9
  }
]
```

### Environment Variables

- `HF_TOKEN`: Hugging Face API token for accessing premium models
- `SESAME_CACHE_DIR`: Custom cache directory for models
- `SESAME_OUTPUT_DIR`: Custom output directory for generated audio

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co/) for transformer models
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Mesop](https://mesop.io/) for the web interface framework

## 📧 Contact

For questions or support, please contact [info@llamasearch.ai](mailto:info@llamasearch.ai)

---

© 2024 LlamaSearch. All Rights Reserved. 
# Updated in commit 1 - 2025-04-04 17:18:57

# Updated in commit 9 - 2025-04-04 17:18:58

# Updated in commit 17 - 2025-04-04 17:18:58

# Updated in commit 25 - 2025-04-04 17:18:59

# Updated in commit 1 - 2025-04-05 14:30:17

# Updated in commit 9 - 2025-04-05 14:30:17

# Updated in commit 17 - 2025-04-05 14:30:17

# Updated in commit 25 - 2025-04-05 14:30:17

# Updated in commit 1 - 2025-04-05 15:16:44

# Updated in commit 9 - 2025-04-05 15:16:44

# Updated in commit 17 - 2025-04-05 15:16:45

# Updated in commit 25 - 2025-04-05 15:16:45

# Updated in commit 1 - 2025-04-05 15:47:28

# Updated in commit 9 - 2025-04-05 15:47:28

# Updated in commit 17 - 2025-04-05 15:47:28
