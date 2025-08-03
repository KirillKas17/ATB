@echo off
echo Запуск модуля эволюции стратегий ATB Trading System...
echo.

REM Активация виртуального окружения
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo Виртуальное окружение активировано
) else (
    echo Виртуальное окружение не найдено, используем системный Python
)

REM Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Ошибка: Python не найден в системе
    pause
    exit /b 1
)

REM Запуск эволюции стратегий
echo Запуск эволюции стратегий...
python scripts/start_evolution.py

if errorlevel 1 (
    echo Ошибка при запуске эволюции стратегий
    pause
    exit /b 1
)

echo Эволюция стратегий завершена
pause 