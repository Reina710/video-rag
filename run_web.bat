@echo off
chcp 65001 >nul
echo ==========================================
echo   启动 video-rag 服务（本地开发）
echo   依赖装好后浏览器打开 http://127.0.0.1:8001
echo   要连中间件请用： docker compose up -d
echo ==========================================
echo.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)
pause
