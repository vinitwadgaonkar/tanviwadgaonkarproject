#!/bin/bash

# Agri Assistant API - Quick Start Script

echo "🚀 Starting Agri Assistant API..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo "📊 Training models..."
    python train_models.py
    echo ""
fi

# Start the API server
echo "🌐 Starting API server at http://127.0.0.1:8000"
echo "📚 API docs available at http://127.0.0.1:8000/docs"
echo "🌍 Open index.html in your browser for the web interface"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py

