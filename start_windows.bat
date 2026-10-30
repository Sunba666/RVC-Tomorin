@echo off
cd /d "%~dp0"
if exist "dist\VoiceWorkbench.exe" (
  start "" "dist\VoiceWorkbench.exe"
) else (
  python main.py
  if errorlevel 1 pause
)
