@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" fractal_explorer.py
if errorlevel 1 pause
