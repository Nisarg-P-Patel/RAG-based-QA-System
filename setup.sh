#!/bin/bash

set -e  # Exit immediately on error

echo "✅ Creating virtual environment..."
python3 -m venv venv

echo "✅ Activating virtual environment..."
source venv/bin/activate

echo "✅ Upgrading pip..."
pip install --upgrade pip

echo "✅ Installing Python dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "❌ requirements.txt not found!"
    exit 1
fi

echo "✅ Downloading GGUF models..."
python models/download_gguf_models.py

# Check if Homebrew is installed
if ! command -v brew &> /dev/null
then
    echo "❌ Homebrew not found. Please install Homebrew first: https://brew.sh/"
    exit 1
fi

brew update

# Install poppler and tesseract
brew install poppler tesseract

echo "✅ Setup completed on macOS."
