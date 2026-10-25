@echo off
chcp 65001 >nul
echo ==========================================
echo   启动 video-rag 服务
echo   启动后浏览器打开 http://127.0.0.1:8001
echo ==========================================
echo.
cd /d "C:\Users\11651\Downloads\video-rag"
"C:\Users\11651\.workbuddy\binaries\python\envs\ca-test\Scripts\python.exe" app.py
pause
