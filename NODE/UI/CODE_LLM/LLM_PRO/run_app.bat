@echo off
chcp 65001 >nul
echo ========================================
echo   Nomos LLM Desktop v1.0
echo ========================================
echo.

cd /d "%~dp0"

REM Python 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python이 설치되어 있지 않습니다.
    echo Python 3.8+ 을 설치해주세요.
    pause
    exit /b 1
)

REM 앱 실행
echo [INFO] 앱을 시작합니다...
python -m app.main

if errorlevel 1 (
    echo.
    echo [ERROR] 앱 실행 중 오류가 발생했습니다.
    echo 의존성을 확인하세요: pip install -r requirements.txt
    pause
)
