@echo off
setlocal ENABLEDELAYEDEXPANSION

set IMAGE_NAME=eggs-webapp
set CONTAINER_NAME=eggs-webapp-container
set PORT=5000

where docker >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Docker не найден. Установите Docker Desktop и запустите этот файл снова.
    pause
    exit /b 1
)

echo [INFO] Собираем Docker-образ (%IMAGE_NAME%)...
docker build -t %IMAGE_NAME% .

for /f "tokens=1" %%i in ('docker ps -a --format "{{.Names}}" ^| findstr /r /c:"^%CONTAINER_NAME%$"') do (
    echo [INFO] Останавливаем старый контейнер %CONTAINER_NAME%...
    docker stop %CONTAINER_NAME% >nul 2>&1
    docker rm %CONTAINER_NAME% >nul 2>&1
)

echo [INFO] Запускаем контейнер...
docker run -d --name %CONTAINER_NAME% -p %PORT%:5000 %IMAGE_NAME%

set URL=http://localhost:%PORT%
echo [INFO] Приложение должно быть доступно по адресу: %URL%

REM Попробуем открыть браузер
start "" %URL%

echo.
echo Нажмите любую клавишу для выхода...
pause >nul
endlocal
