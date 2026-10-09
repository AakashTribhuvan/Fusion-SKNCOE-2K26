@echo off
setlocal
pushd "%~dp0"

set "PROJECT_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%PROJECT_PYTHON%" (
    echo ERROR: Project virtual environment not found.
    echo First create it with: py -3.11 -m venv .venv
    echo Then install dependencies with: .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    popd
    exit /b 1
)

echo Checking the project environment...
call "%~dp0CheckRequirements.bat"
if errorlevel 1 (
    echo.
    echo Setup check failed. Fix the reported issue, then run this file again.
    pause
    popd
    exit /b 1
)

echo.
"%PROJECT_PYTHON%" "%~dp0backend\run_tunnel.py"
if errorlevel 1 pause

popd
exit /b
