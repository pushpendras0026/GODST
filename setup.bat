@echo off
REM setup.bat  --  one-step environment setup for Windows
REM
REM Creates a Python virtual environment in .venv and installs all
REM dependencies listed in requirements.txt.
REM
REM Usage (from this folder):
REM     setup.bat
REM
REM Then activate it with:
REM     .venv\Scripts\activate

setlocal
cd /d "%~dp0"

echo >> Creating virtual environment in .venv ...
python -m venv .venv
if errorlevel 1 goto :error

echo >> Upgrading pip ...
.venv\Scripts\python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo >> Installing requirements ...
.venv\Scripts\pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo >> Done. Activate the environment with:
echo      .venv\Scripts\activate
echo    or run directly, e.g.:
echo      .venv\Scripts\python run.py --stage check
goto :eof

:error
echo.
echo !! Setup failed. Make sure Python 3 is installed and on your PATH.
exit /b 1
