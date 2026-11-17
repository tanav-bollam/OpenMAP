@echo off
rem Launches the Gradio demo with the project's virtual environment.
cd /d "%~dp0"
".venv\Scripts\python.exe" app\demo.py %*
