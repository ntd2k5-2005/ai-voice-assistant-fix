#!/bin/bash
# Setup script for Ollama and Qwen model

set -e

MODEL_NAME=${MODEL_NAME:-"qwen3.5:4b"}

echo "=================================="
echo "Ollama Setup Script"
echo "=================================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed"
    echo ""
    echo "Please install Ollama first:"
    echo "  Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo "  macOS: brew install ollama"
    echo "  Windows: Download from https://ollama.com"
    echo ""
    exit 1
fi

echo "✅ Ollama is installed"
ollama --version
echo ""

# Check if Ollama is running
echo "Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama service is running"
else
    echo "⚠️  Ollama service is not running"
    echo ""
    echo "Starting Ollama service..."
    
    # Try to start Ollama in background
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        ollama serve > /dev/null 2>&1 &
        sleep 3
        
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama service started"
        else
            echo "❌ Failed to start Ollama service"
            echo "Please run 'ollama serve' manually in another terminal"
            exit 1
        fi
    else
        echo "Please run 'ollama serve' manually in another terminal"
        exit 1
    fi
fi
echo ""

# Check if model is already downloaded
echo "Checking for model: $MODEL_NAME"
if ollama list | grep -q "$MODEL_NAME"; then
    echo "✅ Model $MODEL_NAME is already downloaded"
else
    echo "📥 Downloading model: $MODEL_NAME"
    echo "This may take a few minutes..."
    echo ""
    
    if ollama pull "$MODEL_NAME"; then
        echo ""
        echo "✅ Model $MODEL_NAME downloaded successfully"
    else
        echo ""
        echo "❌ Failed to download model: $MODEL_NAME"
        exit 1
    fi
fi
echo ""

# List available models
echo "Available models:"
ollama list
echo ""

echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Install API dependencies:"
echo "     pip install -e \".[api]\""
echo ""
echo "  2. Configure environment:"
echo "     cp .env.example .env"
echo ""
echo "  3. Start the API server:"
echo "     python -m api.main"
echo ""
echo "  4. Test the API:"
echo "     curl http://localhost:8002/health/ready"
echo ""
