@echo off
setlocal

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Project Python environment was not found at:
    echo %PYTHON%
    echo Create it with: py -3.11 -m venv .venv
    echo Then install requirements.txt and rerun this launcher.
    pause
    exit /b 1
)

"%PYTHON%" -c "import fastapi, cv2, tensorflow, torch, faiss, deepface, retinaface, tf_keras, transformers" >nul 2>&1
if errorlevel 1 (
    echo Backend dependencies are missing. Installing them for Python 3.11...
    "%PYTHON%" -m pip install -r "%ROOT%requirements.txt"
    if errorlevel 1 (
        echo Backend dependency installation failed.
        pause
        exit /b 1
    )
)

start "Video Detection Model Backend" /D "%ROOT%" cmd /k ""%PYTHON%" -m uvicorn api.video_app:app --reload --host 127.0.0.1 --port 8000"
start "Video Detection Model Frontend" /D "%ROOT%frontend" cmd /k npm.cmd run dev -- --host 127.0.0.1

echo Video Detection Model screening started.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
endlocal
