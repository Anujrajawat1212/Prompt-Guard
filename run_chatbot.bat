@echo off
title Secure AI Chatbot Launcher
echo ========================================================
echo   Starting Prompt Injection Prevention & Detection Bot
echo ========================================================
echo.

cd /d "%~dp0"

echo [1/2] Checking Python environment...
python -V

echo [2/2] Launching Streamlit Web UI...
python -m streamlit run app.py

pause
