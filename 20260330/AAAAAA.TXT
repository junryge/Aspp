@echo off
title OpenHarness Installer v0.1.0

echo.
echo  ==========================================
echo   OpenHarness Installer v0.1.0
echo  ==========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "OH_HOME=%USERPROFILE%\.openharness"

rem 1. Create user config directory
echo [1/4] Creating config directory: %OH_HOME%
if not exist "%OH_HOME%" mkdir "%OH_HOME%"
if not exist "%OH_HOME%\skills" mkdir "%OH_HOME%\skills"
if not exist "%OH_HOME%\plugins" mkdir "%OH_HOME%\plugins"
if not exist "%OH_HOME%\sessions" mkdir "%OH_HOME%\sessions"
echo       OK

rem 2. Create TOKEN.TXT if not exists
echo [2/4] Checking TOKEN.TXT...
if not exist "%OH_HOME%\TOKEN.TXT" (
    type nul > "%OH_HOME%\TOKEN.TXT"
    echo       Created TOKEN.TXT (empty)
    echo       ** Place your API key in: %OH_HOME%\TOKEN.TXT
) else (
    echo       TOKEN.TXT already exists
)

rem 3. Install package
echo [3/4] Installing OpenHarness...
pip install -e "%SCRIPT_DIR%" >nul 2>&1
if errorlevel 1 (
    echo       pip install failed. Trying --user...
    pip install -e "%SCRIPT_DIR%" --user >nul 2>&1
)
if errorlevel 1 (
    echo       [WARN] pip install failed.
    echo       Use PYTHONPATH method instead:
    echo         set PYTHONPATH=%SCRIPT_DIR%src;%%PYTHONPATH%%
    echo         python -m openharness
) else (
    echo       OK
)

rem 4. Verify
echo [4/4] Verifying installation...
where oh >nul 2>&1
if %errorlevel%==0 (
    echo       [OK] 'oh' command is available
) else (
    echo       [WARN] 'oh' not found in PATH
    echo       Try: python -m openharness
)

echo.
echo  ==========================================
echo   Installation Complete!
echo.
echo   1. Add API key:
echo      notepad %OH_HOME%\TOKEN.TXT
echo.
echo   2. Run:
echo      oh
echo.
echo   3. Help:
echo      oh --help
echo  ==========================================
echo.
pause
