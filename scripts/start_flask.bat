@echo off
REM 🎭 Desi Debate - Hackathon Demo Launcher (Windows)
REM AI-Powered Multi-Agent Debate System

echo.
echo ========================================================
echo 🎭 DESI DEBATE - HACKATHON DEMONSTRATION 🏆
echo ========================================================
echo AI-Powered Multi-Agent Debate System
echo RAG + GNN + RL Integration
echo ========================================================
echo.

REM Check Python
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
) else (
    python --version
    echo ✅ Python OK
)
echo.

REM Quick system test
echo [2/4] Running system validation...
python test_basic_functionality.py >nul 2>&1
if errorlevel 1 (
    echo ⚠️  System validation failed - running setup...
    python hackathon_setup.py
) else (
    echo ✅ System validation passed
)
echo.

REM Check environment
echo [3/4] Checking environment configuration...
if not exist ".env" (
    echo ⚠️  Creating .env file...
    copy .env.example .env >nul 2>&1
    echo ✅ Environment file created
    echo 💡 Tip: Add your OpenAI API key to .env for enhanced features
) else (
    echo ✅ Environment configured
)
echo.

REM Start server
echo [4/4] Starting demonstration server...
echo.
echo 🌐 Demo will be available at: http://localhost:5000
echo 🎯 Try these demo topics:
echo   • Should AI development be regulated by government?
echo   • Is social media's impact on society positive or negative?
echo   • Should universal basic income be implemented?
echo.
echo 🚀 Opening browser and starting server...
echo Press Ctrl+C to stop the demonstration
echo.

REM Open browser
start http://localhost:5000

REM Start Flask
python run_flask.py
pause 