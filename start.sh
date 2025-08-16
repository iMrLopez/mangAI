#!/bin/bash

# MangAI Startup Script
# This script sets up the environment and starts the application

echo "----- APP NEEDS PYTHON 3.11.11 -----"
echo "----- APP NEEDS PYENV -----"

echo "🚀 Starting MangAI Application..."

echo "Setting python version"
pyenv local 3.11.11

echo "Creating venv"
python3 -m venv virtualenv
source virtualenv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p audio_output
mkdir -p logs

# Check if models exist
echo "🔍 Checking models..."
if [ ! -f "./models/yolo8l_50epochs_frame/best.pt" ]; then
    echo "⚠️  Warning: Frame detection model not found at ./models/yolo8l_50epochs_frame/best.pt"
fi

if [ ! -f "./models/yolo8l_50epochs/best.pt" ]; then
    echo "⚠️  Warning: Panel detection model not found at ./models/yolo8l_50epochs/best.pt"
fi

# Check environment variables
echo "🔧 Checking configuration..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ℹ️  OpenAI API key not set - LLM features will use mock data"
fi

# Start the application
echo "🌟 Starting Streamlit application..."
echo "🌐 Application will be available at: http://localhost:8501"
echo ""

# Run Streamlit
exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
