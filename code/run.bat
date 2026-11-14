@echo off
REM Runs a Python script with the project environment (no activation needed).
REM Usage (inside the code folder):  .\run check_env_setup.py
"%~dp0..\venv\Scripts\python.exe" %*
