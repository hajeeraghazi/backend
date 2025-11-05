@echo off
REM Run this from the backend folder. Make sure Python is installed.
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
pause
