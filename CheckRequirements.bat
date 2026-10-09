@echo off
setlocal
pushd "%~dp0"

set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PROJECT_PYTHON%" (
    echo ERROR: Project virtual environment not found.
    echo Create it with: py -3.11 -m venv .venv
    echo Then install dependencies with: .venv\Scripts\python.exe -m pip install -r requirements.txt
    popd
    exit /b 1
)

echo Checking project Python version...
"%PROJECT_PYTHON%" -c "import sys; print('Python', sys.version.split()[0]); sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)"
if errorlevel 1 (
    echo ERROR: This project expects Python 3.11.x in .venv.
    popd
    exit /b 1
)

echo Checking installed package consistency...
"%PROJECT_PYTHON%" -m pip check
if errorlevel 1 (
    echo ERROR: pip found missing or incompatible dependencies.
    popd
    exit /b 1
)

echo Checking required imports...
"%PROJECT_PYTHON%" -c "import fastapi, uvicorn, pydantic_settings, multipart, supabase, mediapipe, cv2, torch, qrcode; print('FastAPI', fastapi.__version__); print('Uvicorn', uvicorn.__version__); print('Supabase', supabase.__version__); print('MediaPipe', mediapipe.__version__); print('OpenCV', cv2.__version__); print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if errorlevel 1 (
    echo ERROR: One or more required imports failed.
    popd
    exit /b 1
)

echo.
echo All declared project requirements are installed and importable.
popd
exit /b 0
