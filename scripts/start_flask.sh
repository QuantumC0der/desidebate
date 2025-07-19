#!/bin/bash

# 🎭 Desi Debate - Hackathon Demo Launcher (Unix/Linux/Mac)
# AI-Powered Multi-Agent Debate System

set -e

echo ""
echo "========================================================"
echo "🎭 DESI DEBATE - HACKATHON DEMONSTRATION 🏆"
echo "========================================================"
echo "AI-Powered Multi-Agent Debate System"
echo "RAG + GNN + RL Integration"
echo "========================================================"
echo ""

# Check Python
echo "[1/4] Checking Python installation..."
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
else
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
    $PYTHON_CMD --version
    echo "✅ Python OK"
fi
echo ""

# Check virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "🔧 Using virtual environment: $(basename $VIRTUAL_ENV)"
else
    echo "💡 Tip: Consider using a virtual environment"
fi

# Quick system test
echo "[2/4] Running system validation..."
if $PYTHON_CMD test_basic_functionality.py > /dev/null 2>&1; then
    echo "✅ System validation passed"
else
    echo "⚠️  System validation failed - running setup..."
    $PYTHON_CMD hackathon_setup.py
fi
echo ""

# Check environment
echo "[3/4] Checking environment configuration..."
if [ ! -f ".env" ]; then
    echo "⚠️  Creating .env file..."
    cp .env.example .env 2>/dev/null || echo "Warning: .env.example not found"
    echo "✅ Environment file created"
    echo "💡 Tip: Add your OpenAI API key to .env for enhanced features"
else
    echo "✅ Environment configured"
fi
echo ""

# Start server
echo "[4/4] Starting demonstration server..."
echo ""
echo "🌐 Demo will be available at: http://localhost:5000"
echo "🎯 Try these demo topics:"
echo "  • Should AI development be regulated by government?"
echo "  • Is social media's impact on society positive or negative?"
echo "  • Should universal basic income be implemented?"
echo ""
echo "🚀 Opening browser and starting server..."
echo "Press Ctrl+C to stop the demonstration"
echo ""

# Try to open browser
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:5000 &
elif command -v open > /dev/null; then
    open http://localhost:5000 &
fi

# Start Flask
$PYTHON_CMD run_flask.py 