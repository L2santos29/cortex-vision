#!/bin/bash
# Cortex-Vision quick start script
set -e

echo "🧠 Cortex-Vision"
echo "=================="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 is required"
    exit 1
fi

# Check venv
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install PyTorch (CPU-only) primero para evitar descargar ~2GB de CUDA
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -q -r requirements.txt

echo ""
echo "🚀 Starting server at http://localhost:8000"
echo "   API docs: http://localhost:8000/docs"
echo ""

python -m src.main
