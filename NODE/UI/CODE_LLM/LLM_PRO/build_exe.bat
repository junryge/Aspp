@echo off
chcp 65001 >nul
echo ========================================
echo   Nomos LLM - EXE 빌드
echo ========================================
echo.

cd /d "%~dp0"

REM PyInstaller 확인
python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [INFO] PyInstaller 설치 중...
    pip install "pyinstaller>=6.0"
)

REM 이전 빌드 정리
echo [1/3] 이전 빌드 정리...
if exist dist\NomosLLM rmdir /s /q dist\NomosLLM
if exist build\nomos rmdir /s /q build\nomos

REM 빌드 실행
echo [2/3] PyInstaller 빌드 시작...
echo       (몇 분 소요될 수 있습니다)
echo.
pyinstaller nomos.spec --noconfirm
if errorlevel 1 (
    echo.
    echo [ERROR] 빌드 실패!
    pause
    exit /b 1
)

REM 배포 폴더 구성
echo.
echo [3/3] 배포 폴더 구성...
if not exist dist\NomosLLM\models mkdir dist\NomosLLM\models
if not exist dist\NomosLLM\aider_projects mkdir dist\NomosLLM\aider_projects

echo.
echo ========================================
echo   빌드 완료!
echo ========================================
echo.
echo   결과: dist\NomosLLM\
echo   실행: dist\NomosLLM\NomosLLM.exe
echo.
echo   [중요] 배포 시 다음 파일을 dist\NomosLLM\ 에 배치:
echo     - models\ 폴더에 GGUF 모델 파일 복사
echo     - token.txt (API 토큰, 선택사항)
echo.
pause
