@echo off
TITLE HR Resume Intelligence Pipeline
COLOR 0A

ECHO ======================================================
ECHO       STARTING HR RESUME SYSTEM PIPELINE
ECHO ======================================================
ECHO.

:: --- STEP 1: INGESTION ---
ECHO [Step 1/2] Scanning for new resumes...
python ingest_resumes.py

IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [ERROR] Ingestion script failed!
    PAUSE
    EXIT /B
)

:: --- STEP 2: DASHBOARD ---
ECHO.
ECHO [Step 2/2] Launching HR Dashboard...
ECHO.
ECHO Opening browser... Press Ctrl+C in this window to stop the server.
streamlit run app.py

PAUSE