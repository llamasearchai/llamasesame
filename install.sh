#!/bin/bash
# LlamaSesame Installation Script

set -e  # Exit on error

echo "========================================"
echo "LlamaSesame Voice Cloning Studio Pro"
echo "Installation Script"
echo "========================================"

# Check Python version
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
    python_version=$($PYTHON_CMD --version | awk '{print $2}')
    echo "Found Python $python_version"
    
    # Check if version is at least 3.8
    if [[ $(echo "$python_version" | cut -d. -f1) -lt 3 || ($(echo "$python_version" | cut -d. -f1) -eq 3 && $(echo "$python_version" | cut -d. -f2) -lt 8) ]]; then
        echo "Error: Python 3.8 or higher is required."
        exit 1
    fi
else
    echo "Error: Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check for pip
if ! command -v pip3 &>/dev/null; then
    echo "Error: pip3 not found. Please install pip for Python 3."
    exit 1
fi

# Check for GPU (CUDA)
if command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected. CUDA will be used for acceleration."
    HAS_GPU=true
else
    echo "No NVIDIA GPU detected. CPU will be used for inference (slower)."
    HAS_GPU=false
fi

# Create virtual environment
echo -e "\nCreating virtual environment..."
$PYTHON_CMD -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "\nUpgrading pip..."
pip install --upgrade pip

# Install torch with appropriate CUDA version if GPU is available
if [ "$HAS_GPU" = true ]; then
    echo -e "\nInstalling PyTorch with CUDA support..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "\nInstalling PyTorch (CPU version)..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install dependencies
echo -e "\nInstalling dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo -e "\nInstalling LlamaSesame..."
pip install -e .

# Ask for HF token
echo -e "\nSome models require authentication with a Hugging Face token."
read -p "Do you want to add a Hugging Face token? (y/n): " ADD_TOKEN
if [[ $ADD_TOKEN == "y" || $ADD_TOKEN == "Y" ]]; then
    read -p "Enter your Hugging Face token: " HF_TOKEN
    echo "HF: $HF_TOKEN" > apikeys.txt
    echo "Token saved to apikeys.txt"
    
    # Also set environment variable
    export HF_TOKEN="$HF_TOKEN"
    
    # Add to shell profile if needed
    read -p "Do you want to add the token to your shell profile? (y/n): " ADD_TO_PROFILE
    if [[ $ADD_TO_PROFILE == "y" || $ADD_TO_PROFILE == "Y" ]]; then
        shell_profile=""
        if [[ -f ~/.bash_profile ]]; then
            shell_profile=~/.bash_profile
        elif [[ -f ~/.profile ]]; then
            shell_profile=~/.profile
        elif [[ -f ~/.bashrc ]]; then
            shell_profile=~/.bashrc
        elif [[ -f ~/.zshrc ]]; then
            shell_profile=~/.zshrc
        fi
        
        if [[ -n $shell_profile ]]; then
            echo -e '\n# Hugging Face token for LlamaSesame\nexport HF_TOKEN="'"$HF_TOKEN"'"' >> "$shell_profile"
            echo "Added HF_TOKEN to $shell_profile"
        else
            echo "Could not find a shell profile to update. Please manually add 'export HF_TOKEN=\"$HF_TOKEN\"' to your shell profile."
        fi
    fi
fi

# Create output directory
echo -e "\nCreating output directory..."
mkdir -p ~/VoiceCloneOutput

# Setup examples
echo -e "\nSetting up examples..."
mkdir -p examples/audio
mkdir -p examples/output
touch examples/audio/.gitkeep
touch examples/output/.gitkeep

# Make example script executable
chmod +x examples/voice_cloning_example.py

echo -e "\n========================================"
echo "Installation completed successfully!"
echo "========================================"
echo -e "\nTo activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo -e "\nTo run the web interface:"
echo "  python app.py"
echo -e "\nTo run the example script:"
echo "  ./examples/voice_cloning_example.py"
echo -e "\nTo use the command line interface:"
echo "  llamasesame --help"
echo -e "\nNote: You need to add sample audio files to examples/audio/ before running the examples."
echo "========================================" 